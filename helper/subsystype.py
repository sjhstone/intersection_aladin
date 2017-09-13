from enum import Enum, unique

@unique
class SubSystemType(Enum):
    """`class` SubSystemType
    ------------------------
    Represents subsystem's type for the NLP problem.
    """
    single = 0
    head = 1
    body = 2
    tail = 3

def get_sub_system_type(sub_sys_count: int, sub_index: int) -> SubSystemType:
    """`function` get_sub_system_type
    ---------------------------------
    Given `M` and `vi`, return type.
    """
    if sub_sys_count > 1:
        if sub_index == 0:
            return SubSystemType.head
        elif sub_index == sub_sys_count-1:
            return SubSystemType.tail
        else:
            return SubSystemType.body
    else:
        return SubSystemType.single
