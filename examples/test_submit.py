#!/usr/bin/env python

import sys, os


if __name__ == "__main__":
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 1

    for i in range(N):
        si = ("_%i" % i) if N > 1 else ""
        this_dir = os.path.dirname(os.path.realpath(__file__))
        slurm = "%s/test%s.sh" % (this_dir, si)
        output = "%s/test%s.out" % (this_dir, si)

        n = 128

        s = []
        s += "#!/bin/sh\n"
        s += "#SBATCH --job-name=H5WTST%s\n" % si
        s += "#SBATCH --ntasks=%i\n" % n
        s += "#SBATCH --cpus-per-task=1\n"
        s += "#SBATCH -p regular\n"
        s += "#SBATCH --output=%s\n" % output
        s += "#SBATCH --exclude=c002\n"
        cmd = ""
        cmd += "mpirun -n %i " % n
        # cmd += "--mca btl_openib_connect_udcm_timeout 5000000 "
        # cmd += "--mca btl ^openib "
        cmd += "%s/test.py %i" % (this_dir, i)
        cmd += "\n"
        s += cmd

        with open(slurm, "w") as f:
            f.writelines(s)

        os.system("sbatch %s" % slurm)

