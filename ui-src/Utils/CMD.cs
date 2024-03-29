﻿using SD_FXUI.Utils;
using System;
using System.IO;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class CMD
    {
        public static async Task ProcessConvertCKPT2Diff(string InputFile, bool emaOnly = false, bool b768 = false, string YamlCfgName = "")
        {
            string WorkDir = FS.GetModelDir() + "onnx\\";
            Host ProcessHost = new Host(WorkDir);
            Host.Print($"\n Startup extract ckpt ({InputFile})..... \n");

            string OutPath = null;
            string AddCmd = "";

            if (InputFile.EndsWith(".safetensors"))
            {
                OutPath = FS.GetModelDir(FS.ModelDirs.Diffusers) + Path.GetFileName(InputFile.Substring(0, InputFile.Length - 12));
                AddCmd = " --from_safetensors";
            }
            else
            {
                OutPath = FS.GetModelDir(FS.ModelDirs.Diffusers) + Path.GetFileName(InputFile.Substring(0, InputFile.Length - 5));
            }

            Directory.CreateDirectory(OutPath);

            if (emaOnly)
            {
                AddCmd += " --extract_ema";
            }

            string YamlCfg = "../../repo/model_data/" + YamlCfgName;

            if (b768)
            {
                AddCmd += " --image_size=768";
                AddCmd += " --prediction_type=v-prediction";
            }

            ProcessHost.Start();
            ProcessHost.Send("\"../../repo/" + PythonEnv.GetPy(Helper.VENV.Any) + "\" \"../../repo/diffusion_scripts/convert/convert_original_stable_diffusion_to_diffusers.py\" " +
                                                                            $"--checkpoint_path=\"{InputFile}\" --dump_path=\"{OutPath}\" " +
                                                                            $"--original_config_file=\"{YamlCfg}\" " + AddCmd);

            ProcessHost.SendExitCommand();

            Host.Print("\n  Extract task is done..... \n");

            Notification.SendNotification("Convertation: ~3min!");
        }
        public static async Task ProcessConvertCKPT2ONNX(string InputFile, bool emaOnly = false, bool b768 = false, string YamlCfgName = "")
        {
            string WorkDir = FS.GetWorkingDir() + "\\repo\\";
            Host ProcessHost = new Host(WorkDir);
            Host.Print($"\n Startup extract ckpt({InputFile})..... \n");

            string OutPath = null;
            string AddCmd = "";

            if (InputFile.EndsWith(".safetensors"))
            {
                OutPath = FS.GetModelDir(FS.ModelDirs.Diffusers) + Path.GetFileName(InputFile.Substring(0, InputFile.Length - 12));
                AddCmd = " --from_safetensors";
            }
            else
            {
                OutPath = FS.GetModelDir(FS.ModelDirs.Diffusers) + Path.GetFileName(InputFile.Substring(0, InputFile.Length - 5));
            }

            Directory.CreateDirectory(OutPath);

            if (emaOnly)
            {
                AddCmd += " --extract_ema";
            }

            string YamlCfg = WorkDir + "/model_data/" + YamlCfgName;

            if (b768)
            {
                AddCmd += " --image_size=768";
                AddCmd += " --prediction_type=v-prediction";
            }

            ProcessHost.Start();
            ProcessHost.Send("\"" + PythonEnv.GetPy(Helper.VENV.Any) + "\" \"./diffusion_scripts/convert/convert_original_stable_diffusion_to_diffusers.py\" " +
                                                                            $"--checkpoint_path=\"{InputFile}\" --dump_path=\"{OutPath}\" " +
                                                                            $"--original_config_file=\"{YamlCfg}\" " + AddCmd);

            string Name = System.IO.Path.GetFileNameWithoutExtension(InputFile);
            if (Name.Length == 0)
            {
                Name = System.IO.Path.GetDirectoryName(InputFile);
            }

            string OutPathONNX = FS.GetModelDir(FS.ModelDirs.ONNX) + Name;
            OutPath = OutPath.Replace("\\", "/");

            ProcessHost.Send("\"" + PythonEnv.GetPy(Helper.VENV.DiffONNX) + "\" \"./diffusion_scripts/convert/convert_diffusers_to_onnx.py\" " +
                                                                            $"--model_path=\"{OutPath}\" --output_path=\"{OutPathONNX}\"");

            ProcessHost.SendExitCommand();

            Notification.SendNotification("Convertation: ~5min!");
        }
        public static async Task ProcessConvertDiff2Onnx(string InputFile)
        {
            Notification.SendNotification("Convertation: ~3min!");
            string WorkDir = FS.GetWorkingDir() + "/repo/";

            Host ProcessHost = new Host(WorkDir, "repo/" + PythonEnv.GetPy(Helper.VENV.DiffONNX));
            Host.Print($"\n Startup extract ckpt({InputFile})..... \n");


            string Name = System.IO.Path.GetFileName(InputFile);
            if (Name.Length == 0)
            {
                Name = System.IO.Path.GetDirectoryName(InputFile);
            }

            string OutPath = FS.GetModelDir(FS.ModelDirs.ONNX) + Name;
            OutPath = OutPath.Replace("\\", "/");
            InputFile = InputFile.Replace("\\", "/");

            Directory.CreateDirectory(OutPath);

            ProcessHost.Start("\"diffusion_scripts/convert/convert_diffusers_to_onnx.py\" " + $"--output_path=\"{OutPath}\"" +
                                                                            $" --model_path=\"{InputFile}\"");

            ProcessHost.SendExitCommand();
            ProcessHost.Wait();

            Host.Print("\n  Extract task is done..... \n");
            Notification.SendNotification("Convertation: done!");
        }

        public static async Task UpscalerRunner(int Size, string File)
        {
            string NewFile = null;
            if (GlobalVariables.EnableGFPGAN)
            {
                string ModelDir = FS.GetModelDir();

                if (!Directory.Exists(ModelDir + "gfpgan\\weights"))
                {
                    Notification.SendNotification("Starting downloading face restoration...");
                }

                Host FaceFixProc = new Host(FS.GetModelDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.Any));
                FaceFixProc.Start($"../repo/diffusion_scripts/modules/inference_gfpgan.py -i {File} -o {FS.GetImagesDir()} -v 1.4 -s 1");

                FaceFixProc.SendExitCommand();
                FaceFixProc.Wait();

                string RestorePath = FS.GetImagesDir() + "restored_imgs\\";
                var Files = FS.GetFilesFrom(RestorePath, new string[] { "png", "jpg" }, false);
                foreach (var file in Files)
                {
                    NewFile = GlobalVariables.ImgPath + Path.GetFileNameWithoutExtension(file) + "_fx.png";
                    System.IO.File.Move(file, NewFile);
                }
                FS.Dir.Delete(RestorePath, true);
                FS.Dir.Delete(FS.GetImagesDir() + "restored_faces\\", true);
                FS.Dir.Delete(FS.GetImagesDir() + "cropped_faces\\", true);
                FS.Dir.Delete(FS.GetImagesDir() + "cmp\\", true);
            }

            string DopCmd = (Size + 1).ToString();
            DopCmd = " -s " + DopCmd;

            if (GlobalVariables.TTA)
                DopCmd += " -x";

            string FileName = FS.GetToolsDir();

            switch (GlobalVariables.CurrentUpscalerType)
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
                case Helper.UpscalerType.WAIFU_CU:
                    {
                        FileName += @"\waifu2x\waifu2x-ncnn-vulkan.exe";
                        DopCmd += $" -n {GlobalVariables.Denoise} -m \"{FS.GetToolsDir() + "waifu2x\\models-cunet"}\" -v ";
                        break;
                    }
                case Helper.UpscalerType.WAIFU_UP_PHOTO:
                    {
                        FileName += @"\waifu2x\waifu2x-ncnn-vulkan.exe";
                        DopCmd += $" -n {GlobalVariables.Denoise} -m \"{FS.GetToolsDir() + "waifu2x\\models-upconv_7_photo"}\" -v ";
                        break;
                    }
                case Helper.UpscalerType.WAIFU_UP_ART:
                    {
                        FileName += @"\waifu2x\waifu2x-ncnn-vulkan.exe";
                        DopCmd += $" -n {GlobalVariables.Denoise} -m \"{FS.GetToolsDir() + "waifu2x\\models-upconv_7_anime_style_art_rgb"}\" -v ";
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
                        DopCmd += $" -n {GlobalVariables.Denoise}";
                        break;
                    }
                default:
                    {
                        if (NewFile != null)
                            GlobalVariables.Form.UpdateViewImg(NewFile);
                    }
                    return;
            }

            if (Size == 0)
                return;

            Host ProcessHost = new Host(FS.GetModelDir(), FileName);
            Host.Print("\n Startup upscale..... \n");

            string OutFile = File.Substring(0, File.Length - 4) + "_upscale.png";
            ProcessHost.Start("-i \"" + File + "\" -o \"" + OutFile + "\"" + DopCmd);
            ProcessHost.Wait();

            if (GlobalVariables.EnableGFPGAN)
            {
                Host ProcesHostTwo = new Host(FS.GetModelDir(), FileName);
                OutFile = NewFile.Substring(0, NewFile.Length - 4) + "_upscale.png";
                ProcesHostTwo.Start("-i \"" + NewFile + "\" -o \"" + OutFile + "\"" + DopCmd);
                ProcesHostTwo.Wait();
                GlobalVariables.Form.UpdateViewImg(OutFile);
            }
            else
            {
                GlobalVariables.Form.UpdateViewImg(OutFile);
            }

            GlobalVariables.Form.InvokeProgressApply();
        }


        public static async Task ProcessConvertVaePt2Diff(string InputFile)
        {
            Notification.SendNotification("Convertation: ~few seconds");
            string WorkDir = FS.GetModelDir() + "vae\\";
            Host ProcessHost = new Host(WorkDir, "repo/" + PythonEnv.GetPy(Helper.VENV.Any));
            Host.Print($"\n Startup convert vae ({InputFile})..... \n");


            string Name = System.IO.Path.GetFileNameWithoutExtension(InputFile);

            string OutPath = WorkDir + Name;
            OutPath = OutPath.Replace("\\", "/");
            InputFile = InputFile.Replace("\\", "/");

            Directory.CreateDirectory(OutPath);

            ProcessHost.Start("\"../../repo/diffusion_scripts/convert/convert_vae_pt_to_diffusers.py\" " + $"--vae_pt_path=\"{InputFile}\"" +
                                                                            $" --dump_path=\"{OutPath + "/vae"}\"");

            ProcessHost.SendExitCommand();
            ProcessHost.Wait();

            Host.Print("\n  Convert task is done..... \n");
            Notification.SendNotification("Convertation: done!");
            GlobalVariables.Form.InvokeUpdateModelsList();
        }

        internal static void ProcessConvertVaePt2ONNX(string InputFile)
        {
            bool NeedNameFix = true;

            if (InputFile.EndsWith("pt"))
            {
                string NewInputFile = System.IO.Path.GetFileNameWithoutExtension(InputFile);
                NewInputFile = FS.GetModelDir() + "vae\\" + NewInputFile;

                if (!Directory.Exists(NewInputFile))
                {
                    ProcessConvertVaePt2Diff(InputFile);
                }
                InputFile = NewInputFile;
                NeedNameFix = false;
            }

            Notification.SendNotification("Convertation: ~few seconds");
            string WorkDir = FS.GetModelDir() + "vae\\";
            Host ProcessHost = new Host(WorkDir, "repo/" + PythonEnv.GetPy(Helper.VENV.Any));
            Host.Print($"\n Startup convert vae ({InputFile})..... \n");

            string OutPath = "";

            if (NeedNameFix)
            {
                string Name = System.IO.Path.GetFileNameWithoutExtension(InputFile);

                OutPath = WorkDir + Name;
            }
            else
            {
                OutPath = InputFile;
            }

            OutPath = OutPath.Replace("\\", "/");
            InputFile = InputFile.Replace("\\", "/");

            Directory.CreateDirectory(OutPath);

            ProcessHost.Start("\"../../repo/diffusion_scripts/convert/convert_vae_pt_to_onnx.py\" " + $"--model_path=\"{InputFile}\"" +
                                                                            $" --output_path=\"{OutPath}\"");

            ProcessHost.SendExitCommand();
            ProcessHost.Wait();

            Host.Print("\n  Convert task is done..... \n");
            Notification.SendNotification("Convertation: done!");

            GlobalVariables.Form.InvokeUpdateModelsList();
        }

        public static async Task DeepDanbooruProcess(string currentImage)
        {
            string DDBModel = FS.GetModelDir() + "deepdanbooru\\model-resnet_custom_v3.pt";
            if (!File.Exists(DDBModel))
            {
                Notification.SendNotification("Starting downloading deepdanbooru model...");
                Directory.CreateDirectory(FS.GetModelDir() + "deepdanbooru\\");

                FileDownloader.DownloadFileAsync("https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt", FS.GetModelDir() + "deepdanbooru\\model-resnet_custom_v3.pt");
                Notification.SendNotification("Downloading deepdanbooru model: done!");
            }

            Host ProcessHost = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.Any));
            Host.Print("\n Processing DeepDanbooru.... \n");
            ProcessHost.Start($"repo/diffusion_scripts/modules/danbooru.py --img=\"{currentImage}\" --model=\"{DDBModel}\"  ");
            GlobalVariables.Form.InvokeProgressUpdate(10);
            ProcessHost.SendExitCommand();
            ProcessHost.Wait();

            Host.Print("\n Processing DeepDanbooru: Done..... \n");
            Notification.SendNotification("Processing DeepDanbooru: Done!");
            GlobalVariables.Form.InvokeProgressUpdate(100);
        }
        public static async Task PoserProcess(string currentImage, ControlNetBase CN)
        {
            CN.CheckCN();

            string OutFile = CN.Outdir() + Path.GetFileNameWithoutExtension(currentImage) + ".png";
            Host ProcessHost = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.Any));
            Host.Print("\n Processing poser.... \n");
            ProcessHost.Start($"repo/diffusion_scripts/controlnet_pipe.py --mode=\"{CN.PreprocessCommandLine()}\" --workdir={FS.GetModelDir() + "controlnet\\huggface\\"} --img=\"{currentImage}\" --model=\"{CN.GetModelPathCN()}\"  --outfile=\"{OutFile}\" ");
            GlobalVariables.Form.InvokeProgressUpdate(10);
            ProcessHost.SendExitCommand();
            ProcessHost.Wait();

            Host.Print("\n Processing poser: Done..... \n");
            Notification.SendNotification("Processing poser: Done!");
            GlobalVariables.Form.InvokeProgressUpdate(100);
        }

        internal static async Task ProcessRunnerDiffCN(string cmdline, int size, ControlNetBase CN)
        {
            CN.CheckSD();

            cmdline += $" --cn_model=\"{CN.GetModelPathSD()}\" ";
            cmdline += $" --outfile=\"{FS.GetWorkingDir()}\" ";

            if (GlobalVariables.Mode == Helper.ImplementMode.ONNX)
            {
                if (!Directory.Exists(FS.GetModelDir(FS.ModelDirs.ONNX) + GlobalVariables.MakeInfo.Model + "\\cnet"))
                {
                    Notification.SendNotification("Your model is old. Reconvert again for use ControlNet API!", true);
                }
            }

            Host ProcessHost = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(GlobalVariables.Mode != Helper.ImplementMode.DiffCUDA ? Helper.VENV.DiffONNX : Helper.VENV.DiffCUDA));
            Host.Print("\n Startup generation..... \n");

            GlobalVariables.Form.InvokeProgressUpdate(7);
            ProcessHost.Start("./repo/diffusion_scripts/controlnet_pipe.py " + cmdline);
            ProcessHost.SendExitCommand();
            GlobalVariables.Form.InvokeProgressUpdate(10);
            ProcessHost.Wait();

            //  process.WaitForInputIdle();
            var Files = FS.GetFilesFrom(FS.GetWorkingDir(), new string[] { "png", "jpg" }, false);
            foreach (var file in Files)
            {
                string NewFilePath = GlobalVariables.ImgPath + System.IO.Path.GetFileName(file);
                System.IO.File.Move(file, NewFilePath);

                await Task.Run(() => UpscalerRunner(size, NewFilePath));
                if (size == 0 || GlobalVariables.CurrentUpscalerType == Helper.UpscalerType.None)
                {
                    GlobalVariables.Form.UpdateViewImg(NewFilePath);
                }
            }

            Host.Print("\n  Task Done..... \n");
            Notification.SendNotification("Task: done!", true);
            GlobalVariables.Form.InvokeProgressUpdate(100);
            GlobalVariables.Form.UpdateCurrentViewImg();
        }
    }
}