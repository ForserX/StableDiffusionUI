using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Shapes;
using System.Xml.Linq;

namespace SD_FXUI
{
    internal class CMD
    {
        public static async Task ProcessConvertCKPT2Diff(string InputFile, bool emaOnly = false)
        {
            string WorkDir = FS.GetModelDir() + "shark\\";
            Host ProcesHost = new Host(WorkDir);
            Host.Print($"\n Startup extract ckpt({InputFile})..... \n");

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

            if (emaOnly)
            {
                AddCmd += " --extract_ema";
            }

            ProcesHost.Start();
            ProcesHost.Send("\"../../repo/" + PythonEnv.GetPy(Helper.VENV.Any) + "\" \"../../repo/diffusion_scripts/convert_original_stable_diffusion_to_diffusers.py\" " +
                                                                            $"--checkpoint_path=\"{InputFile}\" --dump_path=\"{OutPath}\" " +
                                                                            $"--original_config_file=\"../../repo/diffusion_scripts/v1-inference.yaml\" " + AddCmd);

            ProcesHost.SendExitCommand();

            Host.Print("\n  Extract task is done..... \n");

            Wrapper.SendNotification("Convertation: ~3min!");
        }
        public static async Task ProcessConvertCKPT2ONNX(string InputFile, bool emaOnly = false)
        {
            string WorkDir = FS.GetModelDir() + "shark\\";
            Host ProcesHost = new Host(WorkDir);
            Host.Print($"\n Startup extract ckpt({InputFile})..... \n");

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
            
            if (emaOnly)
            {
                AddCmd += " --extract_ema";
            }

            ProcesHost.Start();
            ProcesHost.Send("\"../../repo/" + PythonEnv.GetPy(Helper.VENV.Any) + "\" \"../../repo/diffusion_scripts/convert_original_stable_diffusion_to_diffusers.py\" " +
                                                                            $"--checkpoint_path=\"{InputFile}\" --dump_path=\"{OutPath}\" " +
                                                                            $"--original_config_file=\"../../repo/diffusion_scripts/v1-inference.yaml\" " + AddCmd);

            string Name = System.IO.Path.GetFileNameWithoutExtension(InputFile);
            if (Name.Length == 0)
            {
                Name = System.IO.Path.GetDirectoryName(InputFile);
            }

            string OutPathONNX = FS.GetModelDir() + "onnx\\" + Name;
            OutPath = OutPath.Replace("\\", "/");

            ProcesHost.Send("\"../../repo/" + PythonEnv.GetPy(Helper.VENV.DiffONNX) + "\" \"../../repo/diffusion_scripts/convert_stable_diffusion_checkpoint_to_onnx.py\" " +
                                                                            $"--model_path=\"{OutPath}\" --output_path=\"{OutPathONNX}\"");

            ProcesHost.SendExitCommand();

            Wrapper.SendNotification("Convertation: ~5min!");
        }
        public static async Task ProcessConvertDiff2Onnx(string InputFile)
        {
            Wrapper.SendNotification("Convertation: ~3min!");
            string WorkDir = FS.GetModelDir() + "onnx\\";
            Host ProcesHost = new Host(WorkDir, "repo/" + PythonEnv.GetPy(Helper.VENV.DiffONNX));
            Host.Print($"\n Startup extract ckpt({InputFile})..... \n");


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

            ProcesHost.SendExitCommand();
            ProcesHost.Wait();

            Host.Print("\n  Extract task is done..... \n");
            Wrapper.SendNotification("Convertation: done!");
        }

        public static async Task ProcessRunnerOnnx(string command, int UpSize)
        {
            Host ProcesHost = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.DiffONNX));
            Host.Print("\n Startup generation..... \n");

            Helper.Form.InvokeProgressUpdate(7);
            ProcesHost.Start("./repo/diffusion_scripts/sd_onnx.py " + command);
            ProcesHost.SendExitCommand();
            Helper.Form.InvokeProgressUpdate(10);
            ProcesHost.Wait();

            //  process.WaitForInputIdle();
            var Files = FS.GetFilesFrom(FS.GetWorkingDir(), new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = Helper.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, NewFilePath);

                await Task.Run(() => UpscalerRunner(UpSize, NewFilePath));
                if (UpSize == 0 || Helper.CurrentUpscalerType == Helper.UpscalerType.None)
                {
                    Helper.Form.UpdateViewImg(NewFilePath);
                }
            }

            Host.Print("\n  Task Done..... \n");
            Wrapper.SendNotification("Task: done!");
            Helper.Form.InvokeProgressUpdate(100);
        }
        public static async Task ProcessRunnerDiffCuda(string command, int UpSize)
        {
            Host ProcesHost = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.DiffCUDA));
            Host.Print("\n Startup generation..... \n");

            Helper.Form.InvokeProgressUpdate(7);
            ProcesHost.Start("./repo/diffusion_scripts/sd_diffusers_cuda.py " + command);
            ProcesHost.SendExitCommand();
            Helper.Form.InvokeProgressUpdate(10);
            ProcesHost.Wait();

            //  process.WaitForInputIdle();
            var Files = FS.GetFilesFrom(FS.GetWorkingDir(), new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = Helper.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, NewFilePath);

                await Task.Run(() => UpscalerRunner(UpSize, NewFilePath));
                if (UpSize == 0 || Helper.CurrentUpscalerType == Helper.UpscalerType.None)
                {
                    Helper.Form.UpdateViewImg(NewFilePath);
                }
            }

            Host.Print("\n  Task Done..... \n");
            Wrapper.SendNotification("Task: done!");
            Helper.Form.InvokeProgressUpdate(100);
        }
        public static async Task ProcessRunnerShark(string command, int UpSize)
        {
            Host ProcesHost = new Host(FS.GetModelDir() + "\\shark\\", "repo/" + PythonEnv.GetPy(Helper.VENV.Shark));
            Host.Print("\n Startup generation..... \n");
            Helper.Form.InvokeProgressUpdate(7);

            ProcesHost.Start("../../repo/stable_diffusion/scripts/txt2img.py " + command);
            ProcesHost.SendExitCommand();
            Helper.Form.InvokeProgressUpdate(10);
            ProcesHost.Wait();

            //  process.WaitForInputIdle();
            var Files = FS.GetFilesFrom(FS.GetModelDir() + "\\shark\\", new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = Helper.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, Helper.ImgPath + System.IO.Path.GetFileName(file));

                await Task.Run(() => UpscalerRunner(UpSize, NewFilePath));
                if (UpSize == 0 || Helper.CurrentUpscalerType == Helper.UpscalerType.None)
                {
                    Helper.Form.UpdateViewImg(NewFilePath);
                }
            }

            Host.Print("\n  Task Done..... \n");
            Wrapper.SendNotification("Task: done!");
        }


        public static async Task UpscalerRunner(int Size, string File)
        {
            string NewFile = null;
            if (Helper.EnableGFPGAN)
            {
                Host FaceFixProc = new Host(FS.GetModelDir() + "stable-diffusion//", "repo/" + PythonEnv.GetPy(Helper.VENV.Any));
                FaceFixProc.Start($"../../repo/diffusion_scripts/inference_gfpgan.py -i {File} -o {FS.GetImagesDir()} -v 1.4 -s 1");

                FaceFixProc.SendExitCommand();
                FaceFixProc.Wait();

                string RestorePath = FS.GetImagesDir() + "restored_imgs\\";
                var Files = FS.GetFilesFrom(RestorePath, new string[] { "png", "jpg" }, false);
                foreach (var file in Files)
                {
                    NewFile = Helper.ImgPath + System.IO.Path.GetFileNameWithoutExtension(file) + "_fx.png";
                    System.IO.File.Move(file, NewFile);
                }
                FS.Dir.Delete(RestorePath, true);
                FS.Dir.Delete(FS.GetImagesDir() + "restored_faces\\", true);
                FS.Dir.Delete(FS.GetImagesDir() + "cropped_faces\\", true);
                FS.Dir.Delete(FS.GetImagesDir() + "cmp\\", true);
            }

            string DopCmd = "4";
            DopCmd = " -s " + DopCmd;

            string FileName = FS.GetToolsDir();

            switch (Helper.CurrentUpscalerType)
            {
                case Helper.UpscalerType.ESRGAN:
                    {
                        FileName += @"\realesrgan\realesrgan-ncnn-vulkan.exe";
                        DopCmd += " -v ";
                        break;
                    }
                case Helper.UpscalerType.ESRGAN_X4:
                    {
                        FileName += @"\realesrgan\realesrgan-ncnn-vulkan.exe";
                        DopCmd += " -n realesrgan-x4plus -v ";
                        break;
                    }
                    
                case Helper.UpscalerType.ESRGAN_NET:
                    {
                        FileName += @"\realesrgan\realesrgan-ncnn-vulkan.exe";
                        DopCmd += " -n realesrnet-x4plus -v ";
                        break;
                    }

                case Helper.UpscalerType.ESRGAN_ANIME:
                    {
                        FileName += @"\realesrgan\realesrgan-ncnn-vulkan.exe";

                        DopCmd += " -n realesrgan-x4plus-anime -v ";
                        break;
                    }
                case Helper.UpscalerType.SR:
                    {
                        FileName += @"\realsr\realsr-ncnn-vulkan.exe";
                        break;
                    }
                case Helper.UpscalerType.SRMD:
                    {
                        FileName += @"\srmd\srmd-ncnn-vulkan.exe";
                        break;
                    }
                default:
                    {
                        if (NewFile != null)
                            Helper.Form.UpdateViewImg(NewFile);
                    }
                    return;
            }

            if (Size == 0)
                return;

            Host ProcesHost = new Host(FS.GetModelDir(), FileName);
            Host.Print("\n Startup upscale..... \n");

            string OutFile = File.Substring(0, File.Length - 4) + "_upscale.png";
            ProcesHost.Start("-i " + File + " -o " + OutFile + DopCmd);
            ProcesHost.Wait();

            Helper.Form.UpdateViewImg(OutFile);

            if (Helper.EnableGFPGAN)
            {
                Host ProcesHostTwo = new Host(FS.GetModelDir(), FileName);
                OutFile = NewFile.Substring(0, NewFile.Length - 4) + "_upscale.png";
                ProcesHostTwo.Start("-i " + NewFile + " -o " + OutFile + DopCmd);
                ProcesHostTwo.Wait();
                Helper.Form.UpdateViewImg(OutFile);
            }

            Helper.Form.InvokeProgressApply();
        }
    }
}
