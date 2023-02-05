using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class Install
    {
        public static void Check()
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");

            if (!bDirCheck)
            {
                CMD.InstallApp();
            }
        }
    }
}
