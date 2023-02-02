using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Shapes;

namespace SD_FXUI
{
    internal class CMD
    {
        public static bool TaskDone = false;
        private static String[] GetFilesFrom(String searchFolder, String[] filters, bool isRecursive)
        {
            List<String> filesFound = new List<String>();
            var searchOption = isRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var filter in filters)
            {
                filesFound.AddRange(Directory.GetFiles(searchFolder, String.Format("*.{0}", filter), searchOption));
            }
            return filesFound.ToArray();
        }

        public static async Task ProcessRunner(string command)
        {
            TaskDone = false;

            ProcessStartInfo processStartInfo = new ProcessStartInfo("cmd.exe");
            processStartInfo.RedirectStandardInput = true;
            processStartInfo.WorkingDirectory = System.IO.Directory.GetCurrentDirectory() + "\\models";
            processStartInfo.RedirectStandardOutput = false;

            Process process = Process.Start(processStartInfo);

            if (process != null)
            {
                // FX: Dirty hack for cache 
                string WorkDir = MainWindow.CachePath;

                process.StandardInput.WriteLine(command);
                process.StandardInput.WriteLine("exit");
                process.StandardInput.Flush();
            }

            process.WaitForExit();
            TaskDone = true;

            var Files = GetFilesFrom(System.IO.Directory.GetCurrentDirectory() + "\\models\\", new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = MainWindow.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, MainWindow.ImgPath + System.IO.Path.GetFileName(file));
                MainWindow.Form.UpdateViewImg(NewFilePath);
            }
        }
    }
}
