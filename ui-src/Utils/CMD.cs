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
        public static async Task ProcessRunner(string command, Helper.UpscalerType Type, int UpSize)
        {
            ProcessStartInfo processStartInfo = new ProcessStartInfo("cmd.exe");
            processStartInfo.RedirectStandardInput = true;
            processStartInfo.WorkingDirectory = FS.GetModelDir();
            processStartInfo.RedirectStandardOutput = false;

            Process process = Process.Start(processStartInfo);

            if (process != null)
            {
                // FX: Dirty hack for cache 
                string WorkDir = MainWindow.CachePath;

                process.StandardInput.WriteLine(command);
                process.StandardInput.WriteLine("exit");
                process.StandardInput.Flush();

                process.WaitForExit();

                var Files = FS.GetFilesFrom(FS.GetModelDir(), new string[] { "png", "jpg" }, false);
                foreach (var file in Files)
                {
                    string NewFilePath = MainWindow.ImgPath + System.IO.Path.GetFileName(file);
                    System.IO.File.Move(file, MainWindow.ImgPath + System.IO.Path.GetFileName(file));

                    MainWindow.Form.UpdateViewImg(NewFilePath);

                    if (UpSize> 0)
                    {
                        await UpscalerRunner(Type, UpSize, NewFilePath);
                    }
                }
            }
        }

        public static async Task UpscalerRunner(Helper.UpscalerType Type, int Size, string File)
        {
            Process process = new System.Diagnostics.Process();
            ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();

            string DopCmd = "4";
            DopCmd = " -s " + DopCmd;

            string cmd = FS.GetToolsDir();

            switch (Type)
            {
                case Helper.UpscalerType.ESRGAN:
                    {
                        cmd += @"\realesrgan\realesrgan-ncnn-vulkan.exe";
                        startInfo.FileName = cmd;
                        DopCmd += " -n realesrgan-x4plus";
                        break;
                    }

                case Helper.UpscalerType.ESRGAN_ANIME:
                    {
                        cmd += @"\realesrgan\realesrgan-ncnn-vulkan.exe";
                        startInfo.FileName = cmd;

                        DopCmd += " -n realesrgan-x4plus-anime";
                        break;
                    }
                case Helper.UpscalerType.SR:
                    {
                        cmd += @"\realsr.exe\realsr-ncnn-vulkan.exe";
                        startInfo.FileName = cmd;
                        break;
                    }
                case Helper.UpscalerType.SRMD:
                    {
                        cmd += @"\srmd\srmd-ncnn-vulkan.exe";
                        startInfo.FileName = cmd;
                        break;
                    }
                default:
                    return;
            }

            string OutFile = File.Substring(0, File.Length - 4) + "_upscale.png";
            startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
            startInfo.Arguments = "-i " + File + " -o " + OutFile + DopCmd;
            process.StartInfo = startInfo;
            process.Start();

            process.WaitForExit();
            MainWindow.Form.UpdateViewImg(OutFile);
        }
    }
}
