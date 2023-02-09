using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class PythonEnv
    {
        public static string GetPy(Helper.VENV Type)
        {
            switch (Type)
            {
                case Helper.VENV.DiffCUDA:
                {
                    return "cuda.venv/Scripts/python.exe";
                }
                    
                case Helper.VENV.DiffONNX:
                {
                    return "onnx.venv/Scripts/python.exe";
                }
                    
                case Helper.VENV.Shark:
                {
                    return "shark.venv/Scripts/python.exe";
                }
            }

            string Any = "";
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");

            if(bDirCheck)
            {
                return GetPy(Helper.VENV.Shark);
            }

            bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");

            if (bDirCheck)
            {
                return GetPy(Helper.VENV.DiffONNX);
            }

            bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/cuda.venv");

            if (bDirCheck)
            {
                return GetPy(Helper.VENV.DiffCUDA);
            }

            return Any;
        }

        public static string GetPip(Helper.VENV Type)
        {
            switch (Type)
            {
                case Helper.VENV.DiffCUDA:
                    {
                        return "cuda.venv/Scripts/pip.exe";
                    }

                case Helper.VENV.DiffONNX:
                    {
                        return "onnx.venv/Scripts/pip.exe";
                    }

                case Helper.VENV.Shark:
                    {
                        return "shark.venv/Scripts/pip.exe";
                    }
            }

            string Any = "";
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");

            if (bDirCheck)
            {
                return GetPy(Helper.VENV.Shark);
            }

            bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");

            if (bDirCheck)
            {
                return GetPy(Helper.VENV.DiffONNX);
            }

            bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/cuda.venv");

            if (bDirCheck)
            {
                return GetPy(Helper.VENV.DiffCUDA);
            }

            return Any;
        }
    }
}
