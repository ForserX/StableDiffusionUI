using Microsoft.VisualBasic;
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
        public static async Task ProcessConvertCKPT2Diff(string InputFile)
        {
            string WorkDir = FS.GetModelDir() + "shark\\";
            Host ProcesHost = new Host(WorkDir, "repo/shark.venv/Scripts/python.exe");
            ProcesHost.Print($"\n Startup extract ckpt({InputFile})..... \n");

            if (!File.Exists(WorkDir + "wget.exe"))
            {
                File.Copy(FS.GetToolsDir() + "wget.exe", WorkDir + "wget.exe");
            }

            string OutPath = null;
            string AddCmd = "";

            if (InputFile.EndsWith(".safetensors"))
            {
                OutPath = FS.GetModelDir() + "diff\\" + System.IO.Path.GetFileName(InputFile.Substring(0, InputFile.Length - 12));
                AddCmd = " --from_safetensors";
            }
            else
            {
                OutPath = FS.GetModelDir() + "diff\\" + System.IO.Path.GetFileName(InputFile.Substring(0, InputFile.Length - 5));
            }

            Directory.CreateDirectory(OutPath);

            ProcesHost.Start("\"../../repo/diffusion_scripts/convert_original_stable_diffusion_to_diffusers.py\" " +
                                                                            $"--checkpoint_path=\"{InputFile}\" --dump_path=\"{OutPath}\" " + 
                                                                            $"--original_config_file=\"../../repo/diffusion_scripts/v1-inference.yaml\"" + AddCmd);

            ProcesHost.SendExistCommand();
            ProcesHost.Wait();

            File.Delete(WorkDir + "wget.exe");

            ProcesHost.Print("\n  Extract task is done..... \n");
        }
        public static async Task ProcessConvertDiff2Onnx(string InputFile)
        {
            string WorkDir = FS.GetModelDir() + "onnx\\";
            Host ProcesHost = new Host(WorkDir, "repo/onnx.venv/Scripts/python.exe");
            ProcesHost.Print($"\n Startup extract ckpt({InputFile})..... \n");


            string Name = System.IO.Path.GetFileNameWithoutExtension(InputFile);
            if (Name.Length == 0)
            {
                Name = System.IO.Path.GetDirectoryName(InputFile);
            }

            string OutPath = FS.GetModelDir() + "onnx\\" + Name;
            OutPath = OutPath.Replace("\\", "/");
            InputFile = InputFile.Replace("\\", "/");

            Directory.CreateDirectory(OutPath);

            ProcesHost.Start("\"../../repo/diffusion_scripts/convert_stable_diffusion_checkpoint_to_onnx.py\" " + $"--output_path=\"{OutPath}\"" + 
                                                                            $" --model_path=\"{InputFile}\"");

            ProcesHost.SendExistCommand();
            ProcesHost.Wait();

            ProcesHost.Print("\n  Extract task is done..... \n");
        }

        public static async Task ProcessRunnerOnnx(string command, Helper.UpscalerType Type, int UpSize)
        {
            Host ProcesHost = new Host(FS.GetWorkingDir(), "repo/shark.venv/Scripts/python.exe");
            ProcesHost.Print("\n Startup generation..... \n");

            ProcesHost.Start("./repo/diffusion_scripts/sd_onnx.py " + command);
            ProcesHost.SendExistCommand();
            ProcesHost.Wait();

            //  process.WaitForInputIdle();
            var Files = FS.GetFilesFrom(FS.GetWorkingDir(), new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = Helper.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, Helper.ImgPath + System.IO.Path.GetFileName(file));

                if (UpSize > 0)
                {
                    Task.Run(() => UpscalerRunner(Type, UpSize, NewFilePath));
                }
                else
                {
                    Helper.Form.UpdateViewImg(NewFilePath);
                }
            }

            ProcesHost.Print("\n  Generation Done..... \n");
        }
        public static async Task ProcessRunnerShark(string command, Helper.UpscalerType Type, int UpSize)
        {
            Host ProcesHost = new Host(FS.GetModelDir() + "\\shark\\", "repo/shark.venv/Scripts/python.exe");
            ProcesHost.Print("\n Startup generation..... \n");

            ProcesHost.Start("../../repo/stable_diffusion/scripts/txt2img.py " + command);
            ProcesHost.SendExistCommand();
            ProcesHost.Wait();

            //  process.WaitForInputIdle();
            var Files = FS.GetFilesFrom(FS.GetModelDir() + "\\shark\\", new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = Helper.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, Helper.ImgPath + System.IO.Path.GetFileName(file));

                if (UpSize > 0)
                {
                    Task.Run(() => UpscalerRunner(Type, UpSize, NewFilePath));
                }
                else
                {
                    Helper.Form.UpdateViewImg(NewFilePath);
                }
            }

            ProcesHost.Print("\n  Generation Done..... \n");
        }


        public static async Task UpscalerRunner(Helper.UpscalerType Type, int Size, string File)
        {
            string DopCmd = "4";
            DopCmd = " -s " + DopCmd;

            string FileName = FS.GetToolsDir();

            switch (Type)
            {
                case Helper.UpscalerType.ESRGAN:
                    {
                        FileName += @"\realesrgan\realesrgan-ncnn-vulkan.exe";
                        DopCmd += " -n realesrgan-x4plus";
                        break;
                    }

                case Helper.UpscalerType.ESRGAN_ANIME:
                    {
                        FileName += @"\realesrgan\realesrgan-ncnn-vulkan.exe";

                        DopCmd += " -n realesrgan-x4plus-anime";
                        break;
                    }
                case Helper.UpscalerType.SR:
                    {
                        FileName += @"\realsr.exe\realsr-ncnn-vulkan.exe";
                        break;
                    }
                case Helper.UpscalerType.SRMD:
                    {
                        FileName += @"\srmd\srmd-ncnn-vulkan.exe";
                        break;
                    }
                default:
                    return;
            }

            Host ProcesHost = new Host(FS.GetModelDir(), FileName);
            ProcesHost.Print("\n Startup upscale..... \n");

            string OutFile = File.Substring(0, File.Length - 4) + "_upscale.png";
            ProcesHost.Start("-i " + File + " -o " + OutFile + DopCmd);

            ProcesHost.Wait();
            Helper.Form.UpdateViewImg(OutFile);
        }
    }
}
