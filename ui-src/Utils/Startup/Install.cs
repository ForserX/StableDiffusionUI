using System.IO;

namespace SD_FXUI
{
    internal class Install
    {
        public static void Check()
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");

            if (!bDirCheck)
            {
              //  InstallApp();
            }
        }

        public static void InstallApp()
        {
            //Task.Run(() => InstallGFPGAN());
        }

        public static void CheckAndInstallCUDA(string PyCommand = "py -3.10", bool Silent = false)
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/cuda.venv");
            if (!bDirCheck && (Silent || Notification.MsgBox("Need install packages for use CUDA")))
            {
                Helper.UIHost.Show();
                Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);

                Host.Print("Install CUDA runtimes... Please wait");
                Cmd.Start();
                Cmd.Send(PyCommand + " -m venv .\\repo\\cuda.venv\\");                
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffCUDA)+"\"" + " install -r requirements_cuda.txt ");
                Cmd.SendExitCommand();
                Cmd.Wait();

                if (!Silent)
                {
                    CheckAndInstallCUDA("python", true);
                }
            }

        }
        public static void CheckAndInstallShark(string PyCommand = "py -3.10", bool Silent = false)
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");
            if (!bDirCheck && (Silent || Notification.MsgBox("Need install packages for use Shark")))
            {
                Helper.UIHost.Show();
                Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
                Host.Print("Install python venv... Please wait");
                Cmd.Start();
                Cmd.Send(PyCommand + " -m venv repo\\shark.venv\\");
                Cmd.SendExitCommand();
                Cmd.Wait();

                Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
                Host.Print("Install shark runtimes... Please wait");
                Cmd.Start();
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.Shark)  + "\"" + " install -r requirements_shark.txt");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.Shark)  + "\"" + " install onnxruntime_directml --force");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.Shark)  + "\"" + " --pre torch-mlir torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://llvm.github.io/torch-mlir/package-index/");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.Shark)  + "\"" + " --upgrade -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html iree-compiler iree-runtime");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.Shark)  + "\"" + " -e . -f https://llvm.github.io/torch-mlir/package-index/ -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html");
                Cmd.SendExitCommand();
                Cmd.Wait();

                if (!Silent)
                {
                    CheckAndInstallShark("python", true);
                }
            }

        }

        public static void CheckAndInstallONNX(string PyCommand = "py -3.10", bool Silent = false)
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");
            if (!bDirCheck && (Silent || Notification.MsgBox("Need install packages for use ONNX")))
            {
                Helper.UIHost.Show();
                Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);

                Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
                Host.Print("Install ONNX runtimes... Please wait");
                Cmd.Start();
                Cmd.Send(PyCommand + " -m venv .\\repo\\onnx.venv\\");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffONNX) + "\"" + " install -r requirements_onnx.txt");
                Cmd.SendExitCommand();
                Cmd.Wait();

                if (!Silent)
                {
                    CheckAndInstallONNX("python", true);
                }
            }

        }

        public static void WrapONNXGPU(bool Discrete)
        {

            string FileName = FS.GetWorkingDir() + @"\repo\onnx.venv\Lib\site-packages\diffusers\pipelines\onnx_utils.py";

            if (!System.IO.File.Exists(FileName))
            {
                return;
            }
            using (var reader = System.IO.File.OpenText(FileName))
            {
                int LineCounter = 0;
                string? str = reader.ReadLine();
                while (str != null)
                {
                    if (str.Contains("InferenceSession"))
                    {
                        if (Discrete)
                        {
                            str = "        return ort.InferenceSession(path, providers=[provider], provider_options=[{'device_id': 1}], sess_options=sess_options)";
                        }
                        else
                        {
                            str = "        return ort.InferenceSession(path, providers=[provider], provider_options=[{'device_id': 0}], sess_options=sess_options)";
                        }
                        break;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

                string[] Lines = File.ReadAllLines(FileName);
                Lines[LineCounter] = str;
                System.IO.File.WriteAllLines(FileName, Lines);
            }
        }

        public static void WrapPoserPath()
        {
            string FileName = FS.GetWorkingDir() + @"\repo\onnx.venv\Lib\site-packages\controlnet_aux\open_pose\__init__.py";

            if (Helper.Mode == Helper.ImplementMode.DiffCUDA)
            {
                FileName = FS.GetWorkingDir() + @"\repo\cuda.venv\Lib\site-packages\controlnet_aux\open_pose\__init__.py";
            }

            if (!File.Exists(FileName))
            {
                return;
            }
            using (var reader = System.IO.File.OpenText(FileName))
            {
                int LineCounter = 0;
                string? str = reader.ReadLine();
                while (str != null)
                {
                    if (str.Contains("hf_hub_download(pretrained_model_or_path, filename)"))
                    {
                        str = "        body_model_path = pretrained_model_or_path #hf_hub_download(pretrained_model_or_path, filename)";
                        break;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

                string[] Lines = File.ReadAllLines(FileName);
                Lines[LineCounter] = str;
                System.IO.File.WriteAllLines(FileName, Lines);
            }
        }
        public static void WrapHedPath()
        {
            string FileName = FS.GetWorkingDir() + @"\repo\onnx.venv\Lib\site-packages\controlnet_aux\hed\__init__.py";

            if (Helper.Mode == Helper.ImplementMode.DiffCUDA)
            {
                FileName = FS.GetWorkingDir() + @"\repo\cuda.venv\Lib\site-packages\controlnet_aux\hed\__init__.py";
            }

            if (!File.Exists(FileName))
            {
                return;
            }
            using (var reader = System.IO.File.OpenText(FileName))
            {
                int LineCounter = 0;
                string? str = reader.ReadLine();
                while (str != null)
                {
                    if (str.Contains("hf_hub_download(pretrained_model_or_path, filename)"))
                    {
                        str = "        model_path = pretrained_model_or_path #hf_hub_download(pretrained_model_or_path, filename)";
                        break;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

                string[] Lines = File.ReadAllLines(FileName);
                Lines[LineCounter] = str;
                System.IO.File.WriteAllLines(FileName, Lines);
            }
        }

        public static void WrapMlsdPath()
        {
            string FileName = FS.GetWorkingDir() + @"\repo\onnx.venv\Lib\site-packages\controlnet_aux\mlsd\__init__.py";

            if (Helper.Mode == Helper.ImplementMode.DiffCUDA)
            {
                FileName = FS.GetWorkingDir() + @"\repo\cuda.venv\Lib\site-packages\controlnet_aux\mlsd\__init__.py";
            }

            if (!File.Exists(FileName))
            {
                return;
            }

            using (var reader = System.IO.File.OpenText(FileName))
            {
                int LineCounter = 0;
                string? str = reader.ReadLine();
                while (str != null)
                {
                    if (str.Contains("hf_hub_download(pretrained_model_or_path, filename)"))
                    {
                        str = "        model_path = pretrained_model_or_path #hf_hub_download(pretrained_model_or_path, filename)";
                        break;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

                string[] Lines = File.ReadAllLines(FileName);
                Lines[LineCounter] = str;
                System.IO.File.WriteAllLines(FileName, Lines);
            }
        }

        internal static void SetupDirs()
        {
            Helper.CachePath = FS.GetModelDir() + @"\shark\";
            Helper.ImgPath = FS.GetImagesDir() + System.DateTime.Now.ToString().Replace(':', '-').Replace(' ', '_') + "\\";
            Helper.ImgPath.Replace('\\', '/');

            string OldDiffDirectoryName = FS.GetModelDir() + "diff";
            string NewDiffDirectoryName = FS.GetModelDir(FS.ModelDirs.Diffusers);
            if (Directory.Exists(OldDiffDirectoryName))
            {
                Directory.Move(OldDiffDirectoryName, NewDiffDirectoryName);
            }

            Directory.CreateDirectory(Helper.CachePath);
            Directory.CreateDirectory(FS.GetImagesDir() + "Best");
            Directory.CreateDirectory(FS.GetModelDir() + @"\huggingface");
            Directory.CreateDirectory(FS.GetModelDir() + @"\onnx");
            Directory.CreateDirectory(NewDiffDirectoryName);
            Directory.CreateDirectory(FS.GetModelDir() + @"\vae");
            Directory.CreateDirectory(FS.GetModelDir() + @"\gfpgan");
            Directory.CreateDirectory(FS.GetModelDir() + @"\lora");
            Directory.CreateDirectory(FS.GetModelDir() + @"\textual_inversion");
            Directory.CreateDirectory(FS.GetModelDir() + @"\upscaler");
            Directory.CreateDirectory(FS.GetModelDir() + @"\deepdanbooru");
            Directory.CreateDirectory(FS.GetModelDir() + @"\hypernetwork");
            Directory.CreateDirectory(FS.GetModelDir() + @"\controlnet");
            Directory.CreateDirectory(FS.GetModelDir() + @"\OpenPose");
        }
    }
}
