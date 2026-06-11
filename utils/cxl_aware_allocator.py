def latency_first_allocation(local_budget: int, cxl_budget: int, num_cxl_devices: int, group_items_sizes: dict[int, list[int]]) -> dict[str, tuple[str, list[int]]]:
    allocations = {}
    remaining_local_budget = local_budget
    remaining_cxl_budget = cxl_budget
    level_counts = len(group_items_sizes)
    for level in range(1, level_counts + 1):
        for i, item_size in enumerate(group_items_sizes[level]):
            key = f"level_{level}_item_{i}"
            temp_allocate_on_local = -1
            temp_allocate_on_cxl = -1
            if remaining_local_budget >= item_size:
                local = [1] * 1
                cxls = [0] * num_cxl_devices
                interleave_ratio = local + cxls
                allocations[key] = ("pure_local", interleave_ratio)
                
                temp_allocate_on_local = item_size
                temp_allocate_on_cxl = 0
                remaining_local_budget -= temp_allocate_on_local
                remaining_cxl_budget -= temp_allocate_on_cxl
            elif remaining_local_budget > 0:
                local = [1] * 1
                cxls = [1] * num_cxl_devices
                interleave_ratio = local + cxls
                allocations[key] = ("local_cxl", interleave_ratio)

                temp_allocate_on_local = min(remaining_local_budget, item_size // (1 + num_cxl_devices))
                temp_allocate_on_cxl = item_size - temp_allocate_on_local
                remaining_local_budget -= temp_allocate_on_local
                remaining_cxl_budget -= temp_allocate_on_cxl
            else:
                local = [0] * 1
                cxls = [1] * num_cxl_devices
                interleave_ratio = local + cxls
                allocations[key] = ("pure_cxl", interleave_ratio)
                
                temp_allocate_on_local = 0
                temp_allocate_on_cxl = item_size
                remaining_local_budget -= temp_allocate_on_local
                remaining_cxl_budget -= temp_allocate_on_cxl

    assert remaining_cxl_budget >= 0, "CXL budget exceeded, Run out of memory!"
    return allocations

def convert_interleave_ratio_to_numa_node_mask(interleave_ratio: list[int], local_node_mapping: list[int], cxl_node_mapping: list[int]) -> list[int]:
    # This function converts the interleave ratio to a numa node mask.
    # 1. Convert interleave ratio to numa node mask for numactl
    # - For example, if interleave_ratio = [1, 1] means node 0 and node 1, then the mask is [0, 1]
    # - For example, if interleave_ratio = [1, 0] means only node 0, then the mask is [0]
    # - For example, if interleave_ratio = [0, 1] means only node 1, then the mask is [1]
    # 2. The local_node_mapping and cxl_node_mapping is used to map the interleave_ratio index to actual numa node id.
    # The total number of nodes is len(local_node_mapping) + len(cxl_node_mapping), it must be equal to the length of interleave_ratio. And the order of interleave_ratio is the same as the order of local_node_mapping + cxl_node_mapping.
    # - For example, if local_node_mapping = [0, 1] and cxl_node_mapping = [2, 3], then the interleave_ratio[0] corresponds to numa node 0, interleave_ratio[1] corresponds to numa node 1, interleave_ratio[2] corresponds to numa node 2, and interleave_ratio[3] corresponds to numa node 3.
    # - For example, if local_node_mapping = [0] and cxl_node_mapping = [1], and interleave_ratio = [1, 0], then the mask is [0]
    # - For example, if local_node_mapping = [0] and cxl_node_mapping = [1], and interleave_ratio = [0, 1], then the mask is [1]
    # - For example, if local_node_mapping = [0] and cxl_node_mapping = [1], and interleave_ratio = [1, 1], then the mask is [0, 1] 
    node_mapping = local_node_mapping + cxl_node_mapping
    numa_node_mask = []
    for i, ratio in enumerate(interleave_ratio):
        if ratio == 1:
            numa_node_mask.append(node_mapping[i])
    return numa_node_mask
