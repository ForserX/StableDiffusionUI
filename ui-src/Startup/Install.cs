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

        public static void CheckAndInstallCUDA()
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/cuda.venv");
            if (!bDirCheck && Notification.MsgBox("Need install packages for use CUDA"))
            {
                Helper.UIHost.Show();
                Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);

                Host.Print("Install CUDA runtimes... Please wait");
                Cmd.Start();
                Cmd.Send("python -m venv .\\repo\\cuda.venv\\");                
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffCUDA)+"\"" + " install -r requirements_cuda.txt ");
                Cmd.SendExitCommand();
                Cmd.Wait();
            }

        }
        public static void CheckAndInstallShark()
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");
            if (!bDirCheck && Notification.MsgBox("Need install packages for use Shark"))
            {
                Helper.UIHost.Show();
                Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
                Host.Print("Install python venv... Please wait");
                Cmd.Start();
                Cmd.Send("python -m venv repo\\shark.venv\\");
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

            }

        }

        public static void CheckAndInstallONNX()
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/onnx.venv");
            if (!bDirCheck && Notification.MsgBox("Need install packages for use ONNX"))
            {
                Helper.UIHost.Show();
                Host Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);

                Cmd = new Host(FS.GetWorkingDir(), "cmd.exe", true);
                Host.Print("Install ONNX runtimes... Please wait");
                Cmd.Start();
                Cmd.Send("python -m venv .\\repo\\onnx.venv\\");
                Cmd.Send("\"repo/" + PythonEnv.GetPip(Helper.VENV.DiffONNX) + "\"" + " install -r requirements_onnx.txt");
                Cmd.SendExitCommand();
                Cmd.Wait();
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

                string[] Lines = System.IO.File.ReadAllLines(FileName);
                Lines[LineCounter] = str;
                System.IO.File.WriteAllLines(FileName, Lines);
            }
        }

        internal static void SetupDirs()
        {
            Helper.CachePath = FS.GetModelDir() + @"\shark\";
            Helper.ImgPath = FS.GetImagesDir() + System.DateTime.Now.ToString().Replace(':', '-').Replace(' ', '_') + "\\";
            Helper.ImgPath.Replace('\\', '/');

            System.IO.Directory.CreateDirectory(Helper.CachePath);
            System.IO.Directory.CreateDirectory(Helper.CachePath);
            System.IO.Directory.CreateDirectory(FS.GetImagesDir() + "Best");
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\huggingface");
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\onnx");
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\diff");
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\vae");
        }
    }
}
