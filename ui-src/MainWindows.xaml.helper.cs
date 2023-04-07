using HandyControl.Data;
using SD_FXUI.Properties;
using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace SD_FXUI
{
    public partial class MainWindow
    {
        public void SetPrompt(string Prompt) => CodeUtils.SetRichText(tbPrompt, Prompt);
        public void InvokeSetPrompt(string Prompt) => Dispatcher.Invoke(() => SetPrompt(Prompt));
        public void UpdateViewImg(string Img, bool CN = false) => Dispatcher.Invoke(() => SetImg(Img, CN));
        public void InvokeClearImages() => Dispatcher.Invoke(() => { InvokeClearImages(); });
        public void InvokeProgressUpdate(int value) => Dispatcher.Invoke(() => { pbGen.Value = value; });
        public void InvokeUpdateModelsList() => Dispatcher.Invoke(() => { UpdateModelsList(); });
        public void InvokeUpdateModelsTIList() => Dispatcher.Invoke(() => { UpdateModelsTIList(); });
        public void InvokeProgressApply() => Dispatcher.Invoke(() => { pbGen.Value += (40 / (int)(tbTotalCount.Value)); });
        public void InvokeDropModel() => Dispatcher.Invoke(() => { SafeCMD.Exit(true); });

        public void UpdateCurrentViewImg() => Dispatcher.Invoke(() =>
        {
            if (lvImages.Items.Count > 0)
            {
                int NewIndex = lvImages.Items.Count - 1;

                lvImages.SelectedItem = lvImages.Items[NewIndex];
                lvImages.UpdateLayout();
                try
                {
                    ((ListViewItem)lvImages.ItemContainerGenerator.ContainerFromIndex(NewIndex)).Focus();
                }
                catch
                {
                    Host.Print("Error: WPF lv focus exception!");
                }
            }
        });

        void Load()
        {
            cbFf16.IsChecked = Data.Get("fp16", "true") == "true";
            tsTTA.IsChecked = Data.Get("tta", "false") == "true";
            tsCN.IsChecked = Data.Get("cn", "false") == "true";
            cbGfpgan.IsChecked = Data.Get("cbGfpgan") == "true";
            cbNSFW.IsChecked = Data.Get("cbNSFW") == "true";
            tbH.Text = Data.Get("height", "512");
            tbW.Text = Data.Get("width", "512");
            CodeUtils.SetRichText(tbNegPrompt, Data.Get("neg"));
            tbSteps.Text = Data.Get("steps", "20");
            tbCFG.Text = Data.Get("cfg", "7");
            cbUpscaler.Text = Data.Get("upscaler", "None");
            slUpscale.Value = int.Parse(Data.Get("up_value", "4"));
            cbSampler.Text = Data.Get("sampler", "DDIM");

            Utils.Settings.UseNotif = Data.Get("notif", "true") == "true";
            Utils.Settings.UseNotifImgs = Data.Get("notifi", "true") == "true";
            Utils.Settings.UseInternalVAE = Data.Get("in_vae", "false") == "true";


            bool FirstStart = Data.Get("welcomewnd") != "true";
            if (FirstStart)
            {
                Welcome Hellow = new Welcome();
                Hellow.ShowDialog();

                switch (Helper.Mode)
                {
                    case Helper.ImplementMode.Shark: Helper.Mode = Helper.ImplementMode.IDK; btnShark_Click(0, new RoutedEventArgs()); break;
                    case Helper.ImplementMode.ONNX: Helper.Mode = Helper.ImplementMode.IDK; btnONNX_Click(0, new RoutedEventArgs()); break;
                    case Helper.ImplementMode.DiffCUDA: Helper.Mode = Helper.ImplementMode.IDK; btnDiffCuda_Click(0, new RoutedEventArgs()); break;

                    default: btnONNX_Click(0, new RoutedEventArgs()); break;
                }
            }
            else
            {
                switch (Data.Get("back_mode"))
                {
                    case "2": btnShark_Click(0, new RoutedEventArgs()); break;
                    case "1": btnONNX_Click(0, new RoutedEventArgs()); break;
                    case "3": btnDiffCpu_Click(0, new RoutedEventArgs()); break;
                    case "4": Helper.Mode = Helper.ImplementMode.InvokeAI; break;
                    case "0": btnDiffCuda_Click(0, new RoutedEventArgs()); break;

                    default: btnONNX_Click(0, new RoutedEventArgs()); break;
                }
            }

            UpdateModelsList();
            cbModel.Text = Data.Get("model");
            cbDevice.Text = Data.Get("device");
            cbVAE.Text = Data.Get("VAE");
            cbLoRA.Text = Data.Get("lora");
            cbHyper.Text = Data.Get("hypern");

            if (cbVAE.Text.Length == 0)
            {
                cbVAE.Text = "Default";
            }

            var HistoryStack = Data.Get("history").Split('|');
            foreach (var item in HistoryStack)
            {
                if (item.Length > 0)
                    Helper.PromHistory.Add(item);
            }

            // Disable button for user
            btnDiffCuda.IsEnabled = Helper.GPUID.GreenGPU;
            btnONNX.IsEnabled = Helper.GPUID.RedGPU || Helper.GPUID.NormalBlueGPU;
        }

        void Save()
        {
            Data.Set("fp16", cbFf16.IsChecked == true ? "true" : "false");
            Data.Set("tta", tsTTA.IsChecked == true ? "true" : "false");
            Data.Set("cbGfpgan", cbGfpgan.IsChecked == true ? "true" : "false");
            Data.Set("cn", tsCN.IsChecked == true ? "true" : "false");
            Data.Set("cbNSFW", cbNSFW.IsChecked == true ? "true" : "false");
            Data.Set("notif", Utils.Settings.UseNotif ? "true" : "false");
            Data.Set("notifi", Utils.Settings.UseNotifImgs ? "true" : "false");
            Data.Set("in_vae", Utils.Settings.UseInternalVAE ? "true" : "false");
            Data.Set("height", tbH.Text);
            Data.Set("width", tbW.Text);
            Data.Set("neg", CodeUtils.GetRichText(tbNegPrompt));
            Data.Set("steps", tbSteps.Text);
            Data.Set("upscaler", cbUpscaler.Text);
            Data.Set("up_value", slUpscale.Value.ToString());
            Data.Set("sampler", cbSampler.Text);
            Data.Set("device", cbDevice.Text);
            Data.Set("lora", cbLoRA.Text);
            Data.Set("hypern", cbHyper.Text);
            Data.Set("cfg", tbCFG.Text);

            Data.Set("model", cbModel.Text);
            Data.Set("VAE", cbVAE.Text);
            Data.Set("back_mode", ((int)(Helper.Mode)).ToString());

            string HistoryStack = "";
            foreach (var item in Helper.PromHistory)
            {
                HistoryStack += item + "|";
            }
            Data.Set("history", HistoryStack);

            // Save to file
            Data.Set("welcomewnd", "true");
            Data.Save();
        }

        private string GetCommandLineOnnx()
        {
            string Prompt = CodeUtils.GetRichText(tbPrompt);

            string Model = FS.GetModelDir(FS.ModelDirs.ONNX) + cbModel.Text;

            string VAE = cbVAE.Text.ToLower();
            if (VAE != "default")
            {
                if (VAE.StartsWith("vae\\"))
                {
                    VAE = FS.GetModelDir() + cbVAE.Text.ToLower();
                }
                else
                {
                    VAE = FS.GetModelDir(FS.ModelDirs.ONNX) + cbVAE.Text.ToLower();
                }
            }

            float ETA = float.Parse(tbETA.Text);
            ETA /= 100;
            string newETA = ETA.ToString().Replace(",", ".");
            string CmdLine = $""
                    + $" --prompt=\"{Prompt}\""
                    + $" --prompt_neg=\"{CodeUtils.GetRichText(tbNegPrompt)}\""
                    + $" --height={tbH.Text}"
                    + $" --width={tbW.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --eta={newETA}"
                    + $" --totalcount={tbTotalCount.Value}"
                    + $" --model=\"{Model}\""
                    + $" --vae=\"{VAE}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if (cbNSFW.IsChecked.Value)
            {
                CmdLine += " --nsfw=True";
            }

            if (cbTI.Text != "None" && cbTI.Text.Length > 0)
            {
                CmdLine += $" --inversion={cbTI.Text}";
            }

            if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                float Denoising = float.Parse(tbDenoising.Text);
                Denoising /= 100;

                string newDenoising = Denoising.ToString().Replace(",", ".");

                if (cbPix2Pix.IsChecked.Value)
                {
                    CmdLine += $" --mode=\"pix2pix\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                }
                else
                {
                    if (Helper.ImgMaskPath != string.Empty)
                    {
                        CmdLine += $" --mode=\"inpaint\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                        CmdLine += $" --imgmask=\"{Helper.ImgMaskPath}\"";
                    }
                    else
                    {
                        CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                    }
                }

                if (!File.Exists(Helper.InputImagePath))
                {
                    Notification.MsgBox("Incorrect image path!");
                    CmdLine = "";
                }
            }

            return CmdLine;
        }
        private void MakeCommandObject()
        {
            Helper.MakeInfo.LoRA.Clear();

            string SourcePrompt = CodeUtils.GetRichText(tbPrompt).Replace("\r\n", string.Empty);

            while (SourcePrompt.Contains("<", 0))
            {
                int StartIdx = SourcePrompt.IndexOf("<", 0);
                int EndIdx = SourcePrompt.IndexOf(">", 0) + 1;

                string LoraDataStr = SourcePrompt.Substring(StartIdx, EndIdx - StartIdx);
                SourcePrompt = SourcePrompt.Replace(LoraDataStr, string.Empty);

                LoraDataStr = LoraDataStr.Replace("<", string.Empty).Replace(">", string.Empty);
                if (!LoraDataStr.Contains(":", 0))
                    continue;

                int DelimerIdx = LoraDataStr.IndexOf(":", 0);

                Helper.LoRAData LoRAData = new Helper.LoRAData();
                LoRAData.Name = FS.GetModelDir(FS.ModelDirs.LoRA) + LoraDataStr.Substring(0, DelimerIdx);
                LoRAData.Value = (float)int.Parse(LoraDataStr.Substring(DelimerIdx + 1)) / 100.0f;

                Helper.MakeInfo.LoRA.Add(LoRAData);
            }

            while(SourcePrompt.StartsWith(",") || SourcePrompt.StartsWith(" "))
            {
                SourcePrompt = SourcePrompt.Substring(1);
            }

            Helper.MakeInfo.Prompt = SourcePrompt;
            Helper.MakeInfo.NegPrompt = CodeUtils.GetRichText(tbNegPrompt);
            Helper.MakeInfo.StartSeed = int.Parse(tbSeed.Text);
            Helper.MakeInfo.CFG = float.Parse(tbCFG.Text);
            Helper.MakeInfo.Steps = (int)slSteps.Value;
            Helper.MakeInfo.Model = cbModel.Text;

            if (cbVAE.Text.ToLower() != "default")
            {
                if (cbVAE.Text.StartsWith("vae\\"))
                {
                    Helper.MakeInfo.VAE = FS.GetModelDir() + cbVAE.Text.ToLower();
                }
                else
                {
                    string Dir = (Helper.Mode == Helper.ImplementMode.ONNX) ? "onnx\\" : "diffusers\\";

                    Helper.MakeInfo.VAE = FS.GetModelDir() + Dir + cbVAE.Text.ToLower();
                }
            }
            else
            {
                Helper.MakeInfo.VAE = cbVAE.Text;
            }

            Helper.MakeInfo.Sampler = cbSampler.Text;
            Helper.MakeInfo.ETA = (int)slETA.Value;
            Helper.MakeInfo.TotalCount = (int)tbTotalCount.Value;
            Helper.MakeInfo.Height = (int)slH.Value;
            Helper.MakeInfo.Width = (int)slW.Value;
            Helper.MakeInfo.Device = (Helper.Mode == Helper.ImplementMode.DiffCUDA) ? "cuda" : "cpu";
            Helper.MakeInfo.ImgScale = 0;
            Helper.MakeInfo.WorkingDir = FS.GetWorkingDir();

            if (cbPix2Pix.IsChecked.Value && Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                Helper.MakeInfo.Mode = "pix2pix";
                Helper.MakeInfo.Image = Helper.InputImagePath;
            }
            else if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                Helper.MakeInfo.Mode = "img2img";
                Helper.MakeInfo.Image = Helper.InputImagePath;
                Helper.MakeInfo.ImgScale = (float)slDenoising.Value / 100;
            }
            else if (Helper.DrawMode == Helper.DrawingMode.Inpaint)
            {
                Helper.MakeInfo.Mode = "inpaint";
                Helper.MakeInfo.Image = Helper.InputImagePath;
                Helper.MakeInfo.Mask = Helper.ImgMaskPath;
                Helper.MakeInfo.ImgScale = (float)slDenoising.Value / 100;
            }
            else if (Helper.DrawMode == Helper.DrawingMode.Text2Img)
                Helper.MakeInfo.Mode = "txt2img";

            int Size = (int)slUpscale.Value;
            Helper.CurrentUpscaleSize = Size;

        }

        void ValidateSize()
        {
            if (Helper.Mode != Helper.ImplementMode.ONNX) 
                return;

            if (slH.Value % 128 != 0)
            {
                slH.Value = (double)((int)(slH.Value / 128) * 128);
            }

            if (slW.Value % 128 != 0)
            {
                slW.Value = (double)((int)(slW.Value / 128) * 128);
            }
        }

        private string GetCommandLineDiffCuda()
        {
            string Prompt = CodeUtils.GetRichText(tbPrompt);
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";

            string Model = FS.GetModelDir(FS.ModelDirs.Diffusers) + cbModel.Text;
            if (cbModel.Text.EndsWith(".hgf"))
            {
                Model = cbModel.Text;
                Model = Model.Replace(".hgf", "");
            }

            string VAE = cbVAE.Text.ToLower();
            if (VAE != "default")
            {
                if (VAE.StartsWith("vae\\"))
                {
                    VAE = FS.GetModelDir() + cbVAE.Text.ToLower();
                }
                else
                {
                    VAE = FS.GetModelDir(FS.ModelDirs.Diffusers) + cbVAE.Text.ToLower();
                }
            }

            float ETA = float.Parse(tbETA.Text);
            ETA /= 100;

            string newETA = ETA.ToString().Replace(",", ".");

            string CmdLine = $""
                    + $" --precision={FpMode}"
                    + $" --prompt=\"{Prompt}\""
                    + $" --prompt_neg=\"{CodeUtils.GetRichText(tbNegPrompt)}\""
                    + $" --height={tbH.Text}"
                    + $" --width={tbW.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --image_guidance_scale=\"1.5\""
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --eta={newETA}"
                    + $" --totalcount={tbTotalCount.Value}"
                    + $" --model=\"{Model}\""
                    + $" --vae=\"{VAE}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if (false)
            {
                string HyperModel = FS.GetModelDir() + "hypernetwork\\" + cbHyper.Text;
                CmdLine += $" --hypernetwork=\"{HyperModel}\"";
            }

            if (cbTI.Text != "None" && cbTI.Text.Length > 0)
            {
                CmdLine += $" --inversion={FS.GetModelDir() + "textual_inversion/" + cbTI.Text + ".pt"}";
            }

            if (cbNSFW.IsChecked.Value)
            {
                CmdLine += " --nsfw=True";
            }

            if (CPUUse)
            {
                CmdLine += " --device=\"cpu\"";
            }

            if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                float Denoising = float.Parse(tbDenoising.Text);
                Denoising /= 100;

                string newDenoising = Denoising.ToString().Replace(",", ".");

                if (cbPix2Pix.IsChecked.Value)
                {
                    CmdLine += $" --mode=\"pix2pix\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                }
                else
                {
                    if (Helper.ImgMaskPath != string.Empty)
                    {
                        CmdLine += $" --mode=\"inpaint\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                        CmdLine += $" --imgmask=\"{Helper.ImgMaskPath}\"";
                    }
                    else
                    {
                        CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                    }
                }

                if (!File.Exists(Helper.InputImagePath))
                {
                    Notification.MsgBox("Incorrect image path!");
                    CmdLine = "";
                }
            }

            return CmdLine;
        }
        private string GetCommandLineShark()
        {
            string Prompt = CodeUtils.GetRichText(tbPrompt);

            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string Model = cbModel.Text.EndsWith(".hgf") ? cbModel.Text.Replace(".hgf", "") : FS.GetModelDir(FS.ModelDirs.Diffusers) + cbModel.Text;
            string CmdLine = $" --precision={FpMode}"
                    + $" --device=\"{cbDevice.Text}\""
                    + $" --prompt=\"{Prompt}\""
                    + $" --negative-prompts=\"{CodeUtils.GetRichText(tbNegPrompt)}\""
                    + $" --height={tbH.Text}"
                    + $" --width={tbW.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scheduler={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --total_count={tbTotalCount.Value.ToString()}"
                    + $" --hf_model_id=\"{Model}\""
                    + $" --no-use_tuned "
                    + $" --local_tank_cache=\".//\""
                    + " --enable_stack_trace"
                    //                    + " --iree-vulkan-target-triple=rdna3-unknown-windows"
                    + " --write_metadata_to_png"
            ;

            return CmdLine;
        }

        void SetImg(string Img, bool CN = false)
        {
            ViewImg.Source = CodeUtils.BitmapFromUri(new Uri(Img));
            ListViewItemsCollections.Add(new ListViewItemsData()
            {
                GridViewColumnName_ImageSource = Img,
                GridViewColumnName_LabelContent = "Seed: " + (Helper.MakeInfo.StartSeed + ListViewItemsCollections.Count).ToString()
            });

            lvImages.ItemsSource = ListViewItemsCollections;
            Helper.ImgList.Add(Img);

            btnDDB.Visibility = Visibility.Visible;

            if (CN)
            {
                imgPose.Source = CodeUtils.BitmapFromUri(new Uri(Img));
                Helper.CurrentPose = Img;

                UpdateModelsListControlNet();
                cbPose.SelectedIndex = cbPose.Items.Count - 1;
            }
        }

        public void UpdateModelsTIList()
        {
            string SafeName = (string)cbTI.Text.Clone();
            cbTI.Items.Clear();
            cbTI.Items.Add("None");

            if (Helper.Mode != Helper.ImplementMode.ONNX)
            {
                foreach (string File in Directory.GetFiles(FS.GetModelDir() + "textual_inversion"))
                {
                    cbTI.Items.Add(Path.GetFileNameWithoutExtension(File));
                }
            }
            else
            {
                string Mode = "onnx/";
                string ModelPath = FS.GetModelDir() + Mode + cbModel.Text + "/textual_inversion_merges/";

                if (!Directory.Exists(ModelPath))
                    return;

                foreach (string File in Directory.GetDirectories(ModelPath))
                {
                    cbTI.Items.Add(Path.GetFileNameWithoutExtension(File));
                }
            }

            cbTI.Text = SafeName;
        }

        public void UpdateModelsList()
        {
            string SafeVAE = (string)cbVAE.Text;
            string SafeModelName = (string)cbModel.Text;
            string SafeLoRAName = (string)cbLoRA.Text;
            string SafeHyper = (string)cbHyper.Text;

            cbModel.Items.Clear();
            cbVAE.Items.Clear();
            cbLoRA.Items.Clear();
            cbHyper.Items.Clear();

            cbVAE.Items.Add("Default");
            foreach (var Itm in FS.GetModels(Helper.Mode))
            {
                cbModel.Items.Add(Itm);

                if (!Itm.EndsWith("hgf") && Utils.Settings.UseInternalVAE)
                    cbVAE.Items.Add(Itm);
            }

            // Yeah... LoRA...
            string LoraPath = FS.GetModelDir(FS.ModelDirs.LoRA);
            foreach (var Itm in Directory.GetFiles(LoraPath))
            {
                string TryName = Itm.Replace(LoraPath, string.Empty);

                if (!TryName.EndsWith("txt"))
                    cbLoRA.Items.Add(TryName);
            }

            // Yeah... Hypernetwork...
            string HyperPath = FS.GetModelDir() + "hypernetwork\\";
            foreach (var Itm in Directory.GetFiles(HyperPath))
            {
                string TryName = Itm.Replace(HyperPath, string.Empty);
                cbHyper.Items.Add(TryName);
            }

            foreach (var Itm in Directory.GetDirectories(HyperPath))
            {
                string TryName = Itm.Replace(HyperPath, string.Empty);
                cbHyper.Items.Add(TryName);
            }

            // VAE
            foreach (var Itm in Directory.GetDirectories(FS.GetModelDir() + "vae\\"))
            {
                if (Helper.Mode == Helper.ImplementMode.ONNX)
                {
                    if (!File.Exists(Itm + "\\vae_decoder\\model.onnx"))
                    {
                        continue;
                    }
                }
                else
                {
                    if (!File.Exists(Itm + "\\vae\\diffusion_pytorch_model.bin"))
                    {
                        continue;
                    }
                }

                cbVAE.Items.Add("vae\\" + Path.GetFileName(Itm));
            }

            if (SafeModelName.Length > 0)
            {
                cbModel.Text = SafeModelName;
            }
            else
            {
                cbModel.SelectedIndex = cbModel.Items.Count - 1;
            }

            if (SafeVAE.Length > 0)
            {
                cbVAE.Text = SafeVAE;
            }
            else
            {
                cbVAE.SelectedIndex = 0;
            }

            cbLoRA.Text = SafeLoRAName;
            cbHyper.Text = SafeHyper;
        }

        void ClearImages()
        {
            Helper.ImgList.Clear();
            ViewImg.Source = Helper.NoImageData;

            lvImages.UnselectAll();
            ListViewItemsCollections.Clear();

            Helper.ActiveImageState = Helper.ImageState.Free;
            btnFavor.Source = imgNotFavor.Source;
        }

        private int GCD(int a, int b)
        {
            return b == 0 ? a : GCD(b, a % b);
        }

        private string GetRatio(double Width, double Height)
        {
            int gcd = GCD((int)Width, (int)Height);
            int DeltaWidth = (int)Width / gcd;
            int DeltaHeight = (int)Height / gcd;

            double TotalPix = Width + Height;

            string OutStr = DeltaWidth.ToString() + ":" + DeltaHeight.ToString() + "; " + TotalPix.ToString();

            return OutStr;
        }

        private void ChangeTheme()
        {
            Resources.MergedDictionaries.Clear();
            Resources.MergedDictionaries.Add(new ResourceDictionary
            {
                Source = new Uri($"pack://application:,,,/HandyControl;component/Themes/Skin{SkinType.Violet.ToString()}.xaml")
            });
            Resources.MergedDictionaries.Add(new ResourceDictionary
            {
                Source = new Uri("pack://application:,,,/HandyControl;component/Themes/Theme.xaml")
            });
        }

        private string FixedPrompt(string Text)
        {

            return Text;

        }

        ControlNetBase GetCNType(string ComboBox)
        {
            string LoverName = ComboBox.ToLower();

            if (LoverName == "canny")          return HelperControlNet.Canny;
            if (LoverName == "depth")          return HelperControlNet.Depth;
            if (LoverName == "depth_leres")    ; // not implemented return canny
            if (LoverName == "hed")            return HelperControlNet.Hed;
            if (LoverName == "normalmap")      return HelperControlNet.Normal;
            if (LoverName == "openposedetector")       return HelperControlNet.OpenPose;
            if (LoverName == "openpose_hand")  ; // not implemented return canny
            if (LoverName == "clip_vision")    ; // not implemented return canny
            if (LoverName == "scribble") return HelperControlNet.Scribble;
            if (LoverName == "fake_scribble")  ; // not implemented return canny
            if (LoverName == "pidinet")        ; // not implemented return canny
            if (LoverName == "segmentation") return HelperControlNet.Seg;
            if (LoverName == "mlsd") return HelperControlNet.MLSD;
            if (LoverName == "facegen") return HelperControlNet.Face;
            if (LoverName == "binary")         ; // not implemented return canny

            return HelperControlNet.Canny;                      // temp Bypass error;

        }
        public void UpdateModelsListControlNet()
        {
            cbPose.Items.Clear();
            cbPreprocess.Items.Clear();
            
            if (HelperControlNet.Current == null)
                return;

            string ImgPath = HelperControlNet.Current.Outdir();

            foreach (var Itm in Directory.GetFiles(ImgPath))
            {
                cbPose.Items.Add(Path.GetFileNameWithoutExtension(Itm));
            }

            foreach (var Dir in Directory.GetDirectories(ImgPath))
            {
                foreach (var File in Directory.GetFiles(Dir))
                    cbPose.Items.Add(File.Replace(ImgPath, string.Empty).Replace(".png", string.Empty));
            }

            if (cbPose.Items.Count > 0)
            {
                cbPose.SelectedIndex = 0;
            }
            else
            {
                imgPose.Source = Helper.NoImageData;
            }

            cbPreprocess.Items.Add(HelperControlNet.Current.GetModelName());
            cbPreprocess.SelectedIndex = 0;
        }

        

    }
}
