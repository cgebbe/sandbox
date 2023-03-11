def run_as_cell(name):
    if name != "__main__":
        return False
    if not run_from_ipython():
        return False
    return True


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def setup_ipython():
    if run_from_ipython():
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")

        # get_ipython().system('pip install ipympl')
        # get_ipython().run_line_magic('matplotlib', 'ipympl')


def hi():
    print("hello")
