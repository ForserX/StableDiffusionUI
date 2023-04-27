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
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffCUDA) + "\"" + " install -r requirements_cuda.txt ");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffCUDA) + "\"" + " install protobuf==3.20.* --force");

                Cmd.SendExitCommand();
                Cmd.Wait();

                if (!Silent)
                {
                    CheckAndInstallCUDA("python", true);
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
                Host.Print("Install ONNX runtimes... Please wait");
                Cmd.Start();
                Cmd.Send(PyCommand + " -m venv .\\repo\\onnx.venv\\");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffONNX) + "\"" + " install -r requirements_onnx.txt");

                // Force downgrade
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffONNX) + "\"" + " install ort-nightly-directml==1.15.0.dev20230408001 --force --extra-index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffONNX) + "\"" + " install protobuf==3.20.* --force");

                // Need 2.0.1 for ONNX
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffONNX) + "\"" + " install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu");

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
                string[] Lines = File.ReadAllLines(FileName);

                int LineCounter = 0;
                string? str = reader.ReadLine();
                while (str != null)
                {
                    if (str.Contains("hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)"))
                    {
                        str = "        body_model_path = pretrained_model_or_path + filename";
                        Lines[LineCounter] = str;
                    }

                    if (str.Contains("hf_hub_download(pretrained_model_or_path, hand_filename, cache_dir=cache_dir)"))
                    {
                        str = "        hand_model_path = pretrained_model_or_path + hand_filename";
                        Lines[LineCounter] = str;
                    }

                    if (str.Contains("hf_hub_download(face_pretrained_model_or_path, face_filename, cache_dir=cache_dir)"))
                    {
                        str = "        face_model_path = face_pretrained_model_or_path + face_filename";
                        Lines[LineCounter] = str;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

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
                    if (str.Contains("hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)"))
                    {
                        str = "        model_path = pretrained_model_or_path #hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)";
                        break;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

                if (str == null)
                    return;

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
                    if (str.Contains("hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)"))
                    {
                        str = "        model_path = pretrained_model_or_path #hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)";
                        break;
                    }
                    str = reader.ReadLine();
                    LineCounter++;
                }

                reader.Close();

                if (str == null)
                    return;

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
