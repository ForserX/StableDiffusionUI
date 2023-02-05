using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class Install
    {
        public static void Check()
        {
            bool bDirCheck = System.IO.Directory.Exists(FS.GetWorkingDir() + "/repo/shark.venv");

            if (!bDirCheck)
            {
                InstallApp();
            }
        }

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

            //Task.Run(() => InstallGFPGAN());
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
    }
}
