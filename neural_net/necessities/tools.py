import sys

def progressBar (iteration, total, prefix='Progress', suffix='', rounding=1, l=50, fill='█'):
    percent = ("{0:." + str(rounding) + "f}").format(100 * (iteration / float(total)))
    filledl = int(l * iteration // total)
    bar = fill * filledl + '-' * (l - filledl)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total: print()
