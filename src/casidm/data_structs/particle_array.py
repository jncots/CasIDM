import numpy as np
from enum import Enum
import math


class FilterCode(Enum):
    # xdepth_decay is already calculated
    XD_DECAY_ON = 21
    # xdepth_decay is empty
    XD_DECAY_OFF = 22

class ParticleArray:
    """ParticleArray is data structure to keep particles and their properties

        filter_code: is code to filter entries
        {1: interacting, 2: decaying, 3: final}
    """
    _max_size = 100000000
    _int_type = np.int32
    
    data_attributes = ["pid",
                       "energy",
                       "xdepth",
                       "xdepth_stop",
                       "generation_num",
                       "xdepth_decay",
                       "xdepth_inter",
                       "production_code",
                       "final_code",
                       "filter_code",
                       "id",
                       "parent_id"]

    def __init__(self, size=1000):
        # slice_idx stores advanced indexing array
        self.slice_idx = None
        # Allocate memory
        if size is not None:
            self._allocate(size)
        else:
            # Use is as view
            for attr in self.data_attributes:
                setattr(self, attr, None)
            self._len = None
            self._reserved_size = None
            self.data = None

    def _allocate(self, size):
        self.pid = np.empty(size, dtype=self._int_type)
        self.energy = np.empty(size)
        self.xdepth = np.empty(size)
        self.xdepth_stop = np.empty(size)
        self.generation_num = np.empty(size, dtype=self._int_type)
        self.xdepth_decay = np.empty(size)
        self.xdepth_inter = np.empty(size)
        self.production_code = np.empty(size, dtype=self._int_type)
        self.final_code = np.empty(size, dtype=self._int_type)
        self.filter_code = np.empty(size, dtype=self._int_type)
        self.id = np.empty(size, dtype=np.int64)
        self.parent_id = np.empty(size, dtype=np.int64)
        self.data = self
        self._reserved_size = size
        self._len = 0

    def _adjust_capacity(self, size):
        if size <= self._reserved_size:
            return
        
        old_size = self._reserved_size
        new_size = math.ceil(size/old_size) * old_size
        
        if new_size > self._max_size:
            raise MemoryError("Too large array")
        
        new_array = ParticleArray(new_size)
        for attr in self.data_attributes:
            self_value = getattr(self, attr)
            new_value = getattr(new_array, attr)
            new_value[0:old_size] = self_value
            setattr(self, attr, new_value)
        self._reserved_size = new_size
        
        

    def __len__(self):
        return self._len
    
    
    def __getitem__(self, key):
        view_stack = ParticleArray(None)
        for attr in self.data_attributes:
            value = getattr(self, attr)[key]
            setattr(view_stack, attr, value)

        view_stack._len = view_stack.pid.size
        # view_stack.data = self
        return view_stack
        
    def __setitem__(self, slice_, other):
        """Set arbitrary slice `slice_` of current object `self` with
        values from `other`. Be careful, as it doesn't check whether the
        assignment happens in valid part of the array
        """

        if not isinstance(other, ParticleArray):
            raise ValueError("Argument is not an object of ParticleArray class")
        
        for attr in self.data_attributes:
            other_value = getattr(other, attr)
            self_value = getattr(self, attr)
            self_value[slice_] = other_value
        

    def push(self, **kwargs):
        
        if self._reserved_size is None:
            raise ValueError("A view cannot be extended")

        pid = kwargs.get("pid")
        if pid is None:
            return
        
        # If scalar (one) value is given
        if isinstance(pid, int):
            return self.push_one(**kwargs)
        
        src_slice = kwargs.get("src_slice")
        if src_slice is None:
            src_slice = slice(0, None)
        
        dst_start = self._len
        dst_end = self._len + len(pid[src_slice])
        dst_slice = slice(dst_start, dst_end)
        
        self._adjust_capacity(dst_end)
        
        for name, value in kwargs.items():
            data_attr = getattr(self, name, None)
            if data_attr is not None:
                # Here should be an Exception is thrown
                # If you met a problem here, it means
                # that you gave len(pid) > len(other_data array)
                # Arrays should be equal
                if getattr(value, "__len__", None) is None:
                    data_attr[dst_slice] = value
                else:
                    data_attr[dst_slice] = value[src_slice]
        self._len = dst_end
        return dst_slice
    
    def push_one(self, **kwargs):
        pid = kwargs.get("pid")
        if pid is None:
            return
        
        dst_start = self._len
        dst_end = self._len + 1
        dst_slice = slice(dst_start, dst_end)
        
        self._adjust_capacity(dst_end)
        
        for name, value in kwargs.items():
            data_attr = getattr(self, name, None)
            if data_attr is not None:
                data_attr[dst_slice] = np.copy(value)
        self._len = dst_end
        return dst_slice


    def copy(self, *, src_slice=None, dst_slice=None, size=None):
        if size is None:
            copy_stack = ParticleArray(self._reserved_size)
        else:
            copy_stack = ParticleArray(size)

        if src_slice is None:
            src_slice = slice(0, None)   
            
        copy_stack._len = 0
        if dst_slice is None:
            copy_stack._len = len(self.pid[src_slice])
            dst_slice = slice(0, copy_stack._len)
        
        for attr in self.data_attributes:
            src_value = getattr(self, attr)[src_slice]
            getattr(copy_stack, attr)[dst_slice] = np.copy(src_value)
        
        return copy_stack

    def clear(self, size=None):
        if size is None:
            self._len = 0
        else:
            self._len -= size
            if self._len < 0:
                self._len = 0
        return

    def pop(self, size=None):
        if size is None or self._len < size:
            pop_slice = slice(0, None)
        else:
            pop_slice = slice(self._len - size, self._len)
                      
        popped_stack = self.copy(src_slice=pop_slice)
        self.clear(size)
        return popped_stack
    
    def append(self, other):
        """
        Appends only valid part of other array
        """
        
        if not isinstance(other, ParticleArray):
            raise ValueError("argument is not a ParticleArray object")
        
        len_other = len(other)
        if len_other == 0:
            return self
        
        other_slice =  slice(0, len_other)
        len_self = len(self)
        new_len = len_self + len_other
        self_slice = slice(len_self, new_len)
        
        self._adjust_capacity(new_len)

        for attr in other.data_attributes:
            val_other = getattr(other, attr)
            val_self = getattr(self, attr)
            val_self[self_slice] = val_other[other_slice]
            
        self._len = new_len
        return self    
     
    def refill(self, other):
        self._len = 0
        self.append(other)
            
    def valid(self):
        return self[0:self._len]  

if __name__ == "__main__":
        
    def test_initialization():
        # Initialize stack having size 10 reserved
        
        pstack = ParticleArray(10)

        # Push single values (pid is required)
        print(f"Add single element {pstack.push(pid = 2212)}")
        print(f"Add single element with energy {pstack.push(pid = 2212, energy = 789, xdepth = 0)}")
        # Push arrays
        print("Push array with 3 elements "
            f"{pstack.push(pid = np.array([888, 342, 777]), energy = np.array([20, 20, 20]))}")


        print(pstack.valid().energy)
        
        pstack1 = ParticleArray(10)
        pstack1.push(pid = np.array([888, 342, 777]), energy = 20, xdepth = 13)
        # pstack1.append(pstack).append(pstack)
        print(pstack1.valid().pid)
        print(pstack1.valid().energy)
        print(pstack1.valid().xdepth)
        
    
    
    def test_setitem_function():
        """Test __setitem__ function of ParticleArray
        """
        
        pstack1 = ParticleArray(1)
        pstack1.push(pid = np.array([888, 342, 777]), energy = 20, xdepth = 13)
        
        some_slice = [0, 2]
        pstack2 = pstack1[some_slice]
        print(f"pstack2.pid = {pstack2.pid}")
        
        
        pstack2.pid[0] = 3333
        pstack2.pid[1] = 5555
        pstack2.energy[1] = 77
        # pstack2.push(pid = 999, energy = 888, xdepth = 0)
        print(f"pstack1.pid = {pstack1.pid}")
        print(f"pstack1.energy = {pstack1.energy}")
        print(f"pstack2.pid = {pstack2.pid}")
        print(f"pstack2.energy = {pstack2.energy}")
        print(f"pstack2.valid().pid = {pstack2.valid().pid}")
        print(f"len(pstack2) = {len(pstack2)}")
        # pstack1[some_slice + [6]] = pstack2.valid()
        print(f"pstack1.pid = {pstack1.pid}")
        print(f"pstack1.energy = {pstack1.energy}")
        
    test_setitem_function()     
    
    # print(pstack1.pid)
    # print(len(pstack1))
    # print(pstack1.reserved_size())
    # pstack1 = pstack1._increase_size()
    # pstack1 = pstack1._increase_size()
    # gc.collect()
    # print(len(pstack1))
    # print(pstack1.pid)
    
    
    # pstack1.append(pstack).append(pstack).append(pstack)
    # print(np.where(np.in1d(pstack1.valid().pid, np.array([888])))[0])
    # print(pstack1.valid()[np.where(pstack1.valid().pid in [888, 777])].pid)
    

    # # Check len of stack (number of filled elements)
    # print(f"Size of stack = {len(pstack)}")
    # print(f"Print pid array = {pstack.pid}")
    # print(f"Print data_slice array = {pstack.data_slice}")
    
    # # Pop (remove from stack and return a copy)
    # top_stack = pstack.pop(2)
    # print(f"Size of stack after pop(2) = {len(pstack)}")

    # print(f"Print pid array of pop(2) = {top_stack.pid}")
    # print(f"Print data_slice array of pop(2) = {top_stack.data_slice}")
    # print(f"Size of top_stack = {len(top_stack)}")

# print(len(pstack), pstack.reserved_size())

# print(pstack.pid)
# print(pstack.energy)

# view_p = pstack.view(slice(1, 5))
# print(view_p.pid)

# pstack.pid[4] = 888
# print(view_p.pid, len(view_p))
# print(pstack.pid[np.where(pstack.data_slice>0)])


# print(view_p.data_slice)
# print(view_p.data.view([1, 1, 1]).energy)

# pcopy = pstack.copy(src_slice=np.where((pstack.energy > 1) & (pstack.energy < 1000)),
# dst_slice = slice(1, 5))
# print(pstack.energy)

# print(pcopy.data_slice)
# print(len(pcopy))
# print(pcopy.energy)
