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
                    
                case Helper.VENV.DiffCPU:
                {
                    return "onnx.venv/Scripts/python.exe";
                }

                case Helper.VENV.DiffONNX:
                {
                    return "onnx.venv/Scripts/python.exe";
                }
            }

            string PathForMode = GetPy((Helper.VENV)GlobalVariables.Mode);
            if (System.IO.File.Exists(PathForMode))
            {
                return PathForMode;
            }

            string Any = "";
            int CurModeIdx = (int)GlobalVariables.Mode;
            if (CurModeIdx < 4)
            {
                Any = GetPy((Helper.VENV)CurModeIdx);

                bool DirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + Any);

                if (DirCheck)
                    return Any;
            }

            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");

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
            }

            string PathForMode = GetPip((Helper.VENV)GlobalVariables.Mode);
            if (System.IO.File.Exists(PathForMode))
            {
                return PathForMode;
            }

            string Any = "";
            int CurModeIdx = (int)GlobalVariables.Mode;
            if (CurModeIdx < 4)
            {
                Any = GetPip((Helper.VENV)CurModeIdx);

                bool DirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + Any);

                if (DirCheck)
                    return Any;
            }

            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");

            if (bDirCheck)
            {
                return GetPip(Helper.VENV.DiffONNX);
            }

            bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/cuda.venv");

            if (bDirCheck)
            {
                return GetPip(Helper.VENV.DiffCUDA);
            }

            return Any;
        }

        public static string GetEnv(Helper.VENV Type)
        {
            switch (Type)
            {
                case Helper.VENV.DiffCUDA:
                    {
                        return "cuda.venv";
                    }

                case Helper.VENV.DiffONNX:
                    {
                        return "onnx.venv";
                    }
            }

            string Any = "";
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");

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
