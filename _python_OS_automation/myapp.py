import os
import subprocess

my_env = os.environ.copy()

my_env["PATH"] = os.pathsep.join(["/opt/myapp/", my_env["PATH"]])

result = subprocess.run(["mhapp"], env= my_env)


