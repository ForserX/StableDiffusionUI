using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class HostReader
    {
        static void TorchVersion()
        {
            string WorkDirectory = FS.GetWorkingDir() + "\\repo\\";
            string GetPyExe = WorkDirectory + PythonEnv.GetPy(Helper.VENV.Any);

            string CommandLine = "-c \"import torch; print(torch.__version__)\"";


            Host.Print("Current PyTorch version: ");
            Host ProcessHost = new Host(WorkDirectory, GetPyExe);
            ProcessHost.Start(CommandLine);
        }

        static void PipInstall(string Package)
        {
            string WorkDirectory = FS.GetWorkingDir() + "\\repo\\";
            string GetPyExe = WorkDirectory + PythonEnv.GetPip(Helper.VENV.Any);

            string CommandLine = $"install {Package}";

            Host ProcessHost = new Host(WorkDirectory, GetPyExe);
            ProcessHost.Start(CommandLine);
        }

        public static void Filter(string Message)
        {
            bool Contain = false;
            Host.Print("$> " + Message);

            if (Message.ToLower().Contains("torch -v"))
            {
                TorchVersion();
                Contain = true;
            }

            if (Message.ToLower().Contains("py -install"))
            {
                string Package = Message.Replace("py -install ", string.Empty);

                PipInstall(Package);
                Contain = true;
            }

            if (!Contain)
            {
                Host.Print("Error! Command not found!");
            }
        }
    }
}
