using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Xps.Serialization;

namespace SD_FXUI
{
    internal class ModelCMD
    {
        Host Process = null;
        string Mode = null;
        bool NSFW = false;
        bool CUDA = false;
        bool fp16 = false;
        string Model = null;

        public ModelCMD() 
        {
        }

        public void PreStart(string StartModel, string StartMode, bool StartNSFW, bool IsCUDA = false, bool Infp16 = false)
        {
            if (NSFW != StartNSFW || Model != StartModel || StartMode != Mode || IsCUDA != CUDA || fp16 != Infp16)
            {
                if (Process != null)
                {
                    Exit();
                }

                Mode = StartMode;
                Model = StartModel;
                NSFW = StartNSFW;
                CUDA = IsCUDA;
                fp16 = Infp16;

                string CmdLine;

                if (IsCUDA)
                {
                    CmdLine = $"--model=\"{FS.GetModelDir(FS.ModelDirs.Diffusers) + Model}\" --mode=\"{Mode}\"";
                    CmdLine += $" --precision={(fp16 ? "fp16" : "fp32")}";
                    if (NSFW)
                    {
                        CmdLine += " --nsfw=True ";
                    }

                    if (GlobalVariables.Form.CPUUse)
                    {
                        CmdLine += " --device=\"cpu\"";
                    }
                    else
                    {
                        CmdLine += " --device=\"cuda\"";
                    }

                    Process = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(GlobalVariables.Form.CPUUse ? Helper.VENV.DiffONNX : Helper.VENV.DiffCUDA));
                    Process.Start("./repo/diffusion_scripts/sd_cuda_safe.py " + CmdLine);
                }
                else
                {
                    string ModelFullPath = FS.GetModelDir(FS.ModelDirs.ONNX) + Model;

                    if (Mode == "pix2pix")
                    {
                        ModelFullPath = Model;
                    }

                    CmdLine = $"--model=\"{ModelFullPath}\" --mode=\"{Mode}\"";
                    if (NSFW)
                    {
                        CmdLine += " --nsfw=True ";
                    }

                    Process = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.DiffONNX));
                    Process.Start("./repo/diffusion_scripts/sd_onnx_safe.py " + CmdLine);
                }
            }
            else
            {
                GlobalVariables.Form.InvokeProgressUpdate(20);
            }
        }

        public void Start()
        {
            string JSONStr = JsonConvert.SerializeObject(GlobalVariables.MakeInfo);

            Process.Send(JSONStr);
        }

        public void Exit(bool ErrorExit = false)
        {
            Mode = null;
            NSFW = false;
            CUDA = false;
            fp16 = false;
            Model = null;

            if (ErrorExit)
            {
                if (Process != null)
                {
                    Process.Kill();
                    Process = null;
                }

                return;
            }

            Process.Send("stop");
        }
    }
}