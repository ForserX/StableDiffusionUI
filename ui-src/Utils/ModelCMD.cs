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
        bool LoRA = false;
        string LoRAStr = null;
        string Model = null;
        string ModelLoRA = null;

        public ModelCMD() 
        {
        }

        public void PreStart(string StartModel, string StartMode, bool StartNSFW, string NameLora = null, bool LoraEnable = false, string LoraStrength = null, bool IsCUDA = false, bool Infp16 = false)
        {
            bool LoRACheck = LoRA != LoraEnable;

            if (NSFW != StartNSFW || Model != StartModel || StartMode != Mode || IsCUDA != CUDA || LoRACheck || NameLora != ModelLoRA|| fp16 != Infp16 || LoraStrength != LoRAStr)
            {
                if (Process != null)
                {
                    Exit();
                }

                Mode = StartMode;
                Model = StartModel;
                NSFW = StartNSFW;
                LoRA = LoraEnable;
                CUDA = IsCUDA;
                fp16 = Infp16;
                LoRAStr = LoraStrength;

                string CmdLine;

                if (IsCUDA)
                {
                    CmdLine = $"--model=\"{FS.GetModelDir(FS.ModelDirs.Diffusers) + Model}\" --mode=\"{Mode}\"";
                    CmdLine += $" --precision={(fp16 ? "fp16" : "fp32")}";
                    if (NSFW)
                    {
                        CmdLine += " --nsfw=True ";
                    }

                    if (!LoraEnable)
                    {
                        ModelLoRA = "";
                    }
                    else
                    {
                        ModelLoRA = NameLora;
                    }

                    if (LoraEnable)
                    {

                        float lorastr = float.Parse(LoraStrength);
                        lorastr /= 100;

                        string newLorastr = lorastr.ToString().Replace(",", ".");

                        string LoRAModel = FS.GetModelDir(FS.ModelDirs.LoRA) + ModelLoRA;

                        if (LoRAModel.EndsWith(".safetensors"))
                        {
                            CmdLine += $" --lora=True --lora_path=\"{LoRAModel}\" --lora_strength=\"{newLorastr}\"";
                        }
                        else
                        {
                            CmdLine += $" --dlora=True --lora_path=\"{LoRAModel}\" --lora_strength=\"{newLorastr}\"";
                        }
                    }

                    if (Helper.Form.CPUUse)
                    {
                        CmdLine += " --device=\"cpu\"";
                    }

                    Process = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.Form.CPUUse ? Helper.VENV.DiffONNX : Helper.VENV.DiffCUDA));
                    Process.Start("./repo/diffusion_scripts/sd_cuda_safe.py " + CmdLine);
                }
                else
                {
                    CmdLine = $"--model=\"{FS.GetModelDir(FS.ModelDirs.ONNX) + Model}\" --mode=\"{Mode}\"";
                    if (NSFW)
                    {
                        CmdLine += " --nsfw=True ";
                    }

                    if (!LoraEnable)
                    {
                        ModelLoRA = "";
                    }
                    else
                    {
                        ModelLoRA = NameLora;
                    }

                    if (LoraEnable)
                    {

                        float lorastr = float.Parse(LoraStrength);
                        lorastr /= 100;

                        string newLorastr = lorastr.ToString().Replace(",", ".");

                        string LoRAModel = FS.GetModelDir(FS.ModelDirs.LoRA) + ModelLoRA;

                        if (LoRAModel.EndsWith(".safetensors"))
                        {
                            CmdLine += $" --lora=True --lora_path=\"{LoRAModel}\" --lora_strength=\"{newLorastr}\"";
                        }
                    }

                        Process = new Host(FS.GetWorkingDir(), "repo/" + PythonEnv.GetPy(Helper.VENV.DiffONNX));
                    Process.Start("./repo/diffusion_scripts/sd_onnx_safe.py " + CmdLine);
                }
            }
            else
            {
                Helper.Form.InvokeProgressUpdate(20);
            }
        }

        public void Start()
        {
            string JSONStr = JsonConvert.SerializeObject(Helper.MakeInfo);

            Process.Send(JSONStr);
        }

        public void Exit(bool ErrorExit = false)
        {
            Mode = null;
            NSFW = false;
            CUDA = false;
            fp16 = false;
            LoRA = false;
            Model = null;
            LoRAStr = null;
            ModelLoRA = null;

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