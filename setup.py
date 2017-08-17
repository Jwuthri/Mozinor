"""Module to prepare you python environment."""
import importlib


def run_package(package):
    """Try to run the package just installed."""
    try:
        globals()[package] = importlib.import_module(package)
        print("================================")
    except Exception:
        print("We can't install {}".format(package))
        print("================================")


def install_and_import(package):
    """Install all requiered packages."""
    print("checking for {}".format(str(package)))
    try:
        importlib.import_module(package)
        print("{} is already installed".format(package))
    except ImportError:
        print("We'll install {} before continuing".format(package))
        import pip
        pip.main(['--trusted-host', 'pypi.python.org', 'install', package])
        print("installing {}...".format(package))
    finally:
        run_package(package)


def get_all_packages():
    """"Get all packages in requirement.txt."""
    lst_packages = list()
    with open('requirement.txt') as fp:
        for line in fp:
            lst_packages.append(line.split("=")[0].lower())

    return lst_packages


if __name__ == '__main__':
    lst_install_requires = get_all_packages()
    for module in lst_install_requires:
        install_and_import(module)
    print('You are ready to use the module')
