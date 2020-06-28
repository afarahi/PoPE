import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def main():

    from examples import estimate_TNG_profiles
    import time

    print('START')

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")
        fxn()

        start_time = time.time()
        estimate_TNG_profiles()
        run_time = time.time() - start_time

        print('Run time : ', run_time)

    print('END')


if __name__ == '__main__':
    main()


