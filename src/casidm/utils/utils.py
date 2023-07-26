import os

class suppress_std_streams:
    """Set stdout and stderr to devnull
    
    Taken from https://stackoverflow.com/a/978264
    and https://stackoverflow.com/a/50691621
    """
    def __init__(self, *, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
    
    
    def __enter__(self):        
        if self.suppress_stdout:
            # Open file descriptor
            self.null_stdout = os.open(os.devnull, os.O_RDWR)
            # Save stdout
            self.saved_stdout = os.dup(1)
            # Redirect
            os.dup2(self.null_stdout, 1)
            
        if self.suppress_stderr:
            self.null_stderr = os.open(os.devnull, os.O_RDWR)
            self.saved_stderr =  os.dup(2)
            os.dup2(self.null_stderr, 2)
        return self

    def __exit__(self, *args, **kwargs):
        # restore file descriptors
        if self.suppress_stdout:
            os.dup2(self.saved_stdout, 1)
            os.close(self.null_stdout)
            
        if self.suppress_stderr:
            os.dup2(self.saved_stderr, 2)
            os.close(self.null_stderr)
            
