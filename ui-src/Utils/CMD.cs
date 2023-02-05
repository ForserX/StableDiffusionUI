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
        public static void InstallApp()
        {
            Helper.UIHost.Show();
            Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
            Cmd.Print("Install python venv... Please wait");
            Cmd.Start();
            Cmd.Send("python -m venv repo\\shark.venv\\");
            Cmd.SendExistCommand();

            Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
            Cmd.Print("Install shark runtimes... Please wait");
            Cmd.Start();
            Cmd.Send("repo\\shark.venv\\Scripts\\pip.exe install -r requirements_shark.txt");
            Cmd.Send("repo\\shark.venv\\Scripts\\pip.exe install onnxruntime_directml --force");
            Cmd.Send("repo\\shark.venv\\Scripts\\pip.exe --pre torch-mlir torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://llvm.github.io/torch-mlir/package-index/");
            Cmd.Send("repo\\shark.venv\\Scripts\\pip.exe --upgrade -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html iree-compiler iree-runtime");
            Cmd.Send("repo\\shark.venv\\Scripts\\pip.exe -e . -f https://llvm.github.io/torch-mlir/package-index/ -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html");
            Cmd.SendExistCommand();
            Cmd.Wait();

            Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
            Cmd.Print("Install ONNX runtimes... Please wait");
            Cmd.Start();
            Cmd.Send("python -m venv .\\repo\\onnx.venv\\");
            Cmd.Send("repo\\onnx.venv\\Scripts\\pip.exe install -r requirements_onnx.txt");
            Cmd.SendExistCommand();
            Cmd.Wait();

            Task.Run(() => CMD.InstallGFPGAN());
        }

        public static async Task InstallGFPGAN()
        {
            string WorkDir = FS.GetModelDir() + "onnx\\";
            string WGet = FS.GetToolsDir() + "wget.exe";
            Host WGetProc = new Host(WorkDir, WGet);

            WGetProc.Start("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth");
            WGetProc.SendExistCommand();
            WGetProc.Wait();


            Host ProcesHost = new Host(WorkDir, "repo/onnx.venv/Scripts/python.exe", true);
            ProcesHost.Start($"../../repo/diffusion_scripts/torch2onnx.py --src_model_path=\"{WorkDir}GFPGANv1.3.pth\" --dst_model_path=\"{WorkDir}GFPGANv1.3.onnx\" --img_size=512");

            ProcesHost.SendExistCommand();
            ProcesHost.Wait();

            File.Delete(WorkDir + "GFPGANv1.3.pth");

            Helper.UIHost.Hide();
        }

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
