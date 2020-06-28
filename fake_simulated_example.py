import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def main():
    print('START')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

        import time
        from examples import controled_experiment

        start_time = time.time()
        controled_experiment()
        run_time = time.time() - start_time

        print('Run time : ', run_time)

    print('END')

if __name__ == '__main__':
    main()