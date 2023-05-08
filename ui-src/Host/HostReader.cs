using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class HostReader
    {
        static void PackageVersion(string PackageName)
        {
            PackageName = PackageName.ToLower();

            string WorkDirectory = FS.GetWorkingDir() + "\\repo\\";
            string GetPyExe = WorkDirectory + PythonEnv.GetPy(Helper.VENV.Any);

            string CommandLine = $"-c \"import {PackageName}; print({PackageName}.__version__)\"";


            Host.Print($"Current {CodeUtils.ToUpperFirstLetter(PackageName)} version: ");
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

        static void PipUninstall(string Package)
        {
            string WorkDirectory = FS.GetWorkingDir() + "\\repo\\";
            string GetPyExe = WorkDirectory + PythonEnv.GetPip(Helper.VENV.Any);

            string CommandLine = $"uninstall {Package}";

            Host ProcessHost = new Host(WorkDirectory, GetPyExe);
            ProcessHost.Start(CommandLine);
        }

        public static void Filter(string Message)
        {
            bool Contain = false;
            Host.Print("$> " + Message);

            if (Message.ToLower().Contains("-v"))
            {
                var Words = Message.Split(' ');
                Contain = Words.Length > 2;

                if (Contain)
                {
                    PackageVersion(Words[2]);
                }
            }

            if (Message.ToLower().Contains("py -install"))
            {
                string Package = Message.Replace("py -install ", string.Empty);

                PipInstall(Package);
                Contain = true;
            }

            if (Message.ToLower().Contains("py -uninstall"))
            {
                string Package = Message.Replace("py -uninstall ", string.Empty);

                PipUninstall(Package);
                Contain = true;
            }
            

            if (Message.Length < 8 && Message.ToLower().Contains("clear"))
            {
                Helper.UIHost.Clear();
                Contain = true;
            }

            if (!Contain)
            {
                Host.Print("Error! Command not found!");
            }
        }
    }
}
