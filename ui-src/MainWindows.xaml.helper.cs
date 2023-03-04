﻿using HandyControl.Data;
using SD_FXUI.Properties;
using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace SD_FXUI
{
    public partial class MainWindow
    {
        public void SetPrompt(string Prompt) => TryPrompt.Text = Prompt;
        public void InvokeSetPrompt(string Prompt) => Dispatcher.Invoke(() => SetPrompt(Prompt));
        public void UpdateViewImg(string Img) => Dispatcher.Invoke(() => SetImg(Img));
        public void InvokeClearImages() => Dispatcher.Invoke(() => { InvokeClearImages(); });
        public void InvokeProgressUpdate(int value) => Dispatcher.Invoke(() => { pbGen.Value = value; });
        public void InvokeUpdateModelsList() => Dispatcher.Invoke(() => { UpdateModelsList(); });
        public void InvokeProgressApply() => Dispatcher.Invoke(() => { pbGen.Value += (40 / (int)(tbTotalCount.Value)); });
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
            cbGfpgan.IsChecked = Data.Get("cbGfpgan") == "true";
            cbNSFW.IsChecked = Data.Get("cbNSFW") == "true";
            tbH.Text = Data.Get("height", "512");
            tbW.Text = Data.Get("width", "512");
            NegPrompt.Text = Data.Get("neg");
            tbSteps.Text = Data.Get("steps", "20");
            tbCFG.Text = Data.Get("cfg", "7");
            cbUpscaler.Text = Data.Get("upscaler", "None");
            slUpscale.Value = int.Parse(Data.Get("up_value", "4"));
            cbSampler.Text = Data.Get("sampler", "DDIM");
            tsLoRA.IsChecked = Data.Get("lora_enable", "false") == "true";

            Utils.Settings.UseNotif = Data.Get("notif", "true") == "true";
            Utils.Settings.UseNotifImgs = Data.Get("notifi", "true") == "true";

            switch (Data.Get("back_mode"))
            {
                case "2": btnShark_Click(0, new RoutedEventArgs()); break;
                case "1": btnONNX_Click(0, new RoutedEventArgs()); break;
                case "3": btnDiffCpu_Click(0, new RoutedEventArgs()); break;
                case "4": Helper.Mode = Helper.ImplementMode.InvokeAI; break;
                case "0": btnDiffCuda_Click(0, new RoutedEventArgs()); break;

                default: btnONNX_Click(0, new RoutedEventArgs()); break;
            }

            UpdateModelsList();
            cbModel.Text = Data.Get("model");
            cbDevice.Text = Data.Get("device");
            cbVAE.Text = Data.Get("VAE");
            cbLoRA.Text = Data.Get("lora");

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


            bool FirstStart = Data.Get("welcomewnd") != "true";
            if (FirstStart)
            {
                Welcome Hellow = new Welcome();
                Hellow.ShowDialog();
            }
        }

        void Save()
        {
            Data.Set("fp16", cbFf16.IsChecked == true ? "true" : "false");
            Data.Set("lora_enable", tsLoRA.IsChecked == true ? "true" : "false");
            Data.Set("cbGfpgan", cbGfpgan.IsChecked == true ? "true" : "false");
            Data.Set("cbNSFW", cbNSFW.IsChecked == true ? "true" : "false");
            Data.Set("notif", Utils.Settings.UseNotif ? "true" : "false");
            Data.Set("notifi", Utils.Settings.UseNotifImgs ? "true" : "false");
            Data.Set("height", tbH.Text);
            Data.Set("width", tbW.Text);
            Data.Set("neg", NegPrompt.Text);
            Data.Set("steps", tbSteps.Text);
            Data.Set("upscaler", cbUpscaler.Text);
            Data.Set("up_value", slUpscale.Value.ToString());
            Data.Set("sampler", cbSampler.Text);
            Data.Set("device", cbDevice.Text);
            Data.Set("lora", cbLoRA.Text);
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
            string Prompt = FixedPrompt(TryPrompt.Text);

            string Model = FS.GetModelDir() + "onnx\\" + cbModel.Text;

            string VAE = cbVAE.Text.ToLower();
            if (VAE != "default")
            {
                if (VAE.StartsWith("vae\\"))
                {
                    VAE = FS.GetModelDir() + cbVAE.Text.ToLower();
                }
                else
                {
                    VAE = FS.GetModelDir() + "onnx\\" + cbVAE.Text.ToLower();
                }
            }

            float ETA = float.Parse(tbETA.Text);
            ETA /= 100;
            string newETA = ETA.ToString().Replace(",", ".");
            string CmdLine = $""
                    + $" --prompt=\"{Prompt}\""
                    + $" --prompt_neg=\"{NegPrompt.Text}\""
                    + $" --height={tbH.Text}"
                    + $" --width={tbW.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --eta={newETA}"
                    + $" --totalcount={tbTotalCount.Value.ToString()}"
                    + $" --model=\"{Model}\""
                    + $" --vae=\"{VAE}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if (cbNSFW.IsChecked.Value)
            {
                CmdLine += " --nsfw=True";
            }

            if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                float Denoising = float.Parse(tbDenoising.Text);
                Denoising /= 100;

                string newDenoising = Denoising.ToString().Replace(",", ".");

                if(Helper.ImgMaskPath != string.Empty)
                {
                    CmdLine += $" --mode=\"inpaint\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                    CmdLine += $" --imgmask=\"{Helper.ImgMaskPath}\"";
                }
                else
                {
                    CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";
                }

                if (!File.Exists(Helper.InputImagePath))
                {
                    Notification.MsgBox("Incorrect image path!");
                    CmdLine = "";
                }
            }

            return CmdLine;
        }
        private string GetCommandLineDiffCuda()
        {
            string Prompt = FixedPrompt(TryPrompt.Text);
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";

            string Model = string.Empty;
            if (cbModel.Text.EndsWith(".hgf"))
            {
                Model = cbModel.Text;
                Model = Model.Replace(".hgf", "");
            }
            else
            {
                Model = FS.GetModelDir() + "diffusers\\" + cbModel.Text;
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
                    VAE = FS.GetModelDir() + "diffusers\\" + cbVAE.Text.ToLower();
                }
            }

            float ETA = float.Parse(tbETA.Text);
            ETA /= 100;

            string newETA = ETA.ToString().Replace(",", ".");

            string CmdLine = $""
                    + $" --precision={FpMode}"
                    + $" --prompt=\"{Prompt}\""
                    + $" --prompt_neg=\"{NegPrompt.Text}\""
                    + $" --height={tbH.Text}"
                    + $" --width={tbW.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --eta={newETA}"
                    + $" --totalcount={tbTotalCount.Value.ToString()}"
                    + $" --model=\"{Model}\""
                    + $" --vae=\"{VAE}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if (tsLoRA.IsChecked.Value)
            {
                string LoRAModel = FS.GetModelDir() + "lora\\" + cbLoRA.Text;
                CmdLine += $" --lora=True --lora_path=\"{LoRAModel}\"";
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

                CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale={newDenoising}";

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
            string Prompt = FixedPrompt(TryPrompt.Text);

            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string Model = cbModel.Text.EndsWith(".hgf") ? cbModel.Text.Replace(".hgf", "") : FS.GetModelDir() + "diffusers\\" + cbModel.Text;
            string CmdLine = $" --precision={FpMode}"
                    + $" --device=\"{cbDevice.Text}\""
                    + $" --prompt=\"{Prompt}\""
                    + $" --negative-prompts=\"{NegPrompt.Text}\""
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

        void SetImg(string Img)
        {
            ViewImg.Source = new BitmapImage(new Uri(Img));
            ListViewItemsCollections.Add(new ListViewItemsData()
            {
                GridViewColumnName_ImageSource = Img
            });

            lvImages.ItemsSource = ListViewItemsCollections;
            Helper.ImgList.Add(Img);

            btnDDB.Visibility = Visibility.Visible;
        }

        public void UpdateModelsList()
        {
            cbModel.Items.Clear();
            cbVAE.Items.Clear();
            cbLoRA.Items.Clear();

            cbVAE.Items.Add("Default");
            foreach (var Itm in FS.GetModels(Helper.Mode))
            {
                cbModel.Items.Add(Itm);

                if (!Itm.EndsWith("hgf"))
                    cbVAE.Items.Add(Itm);
            }

            // Yeah... LoRA...
            string LoraPath = FS.GetModelDir() + "lora\\";
            foreach (var Itm in Directory.GetFiles(LoraPath))
            {
                string TryName = Itm.Replace(LoraPath, string.Empty);
                cbLoRA.Items.Add(TryName);
            }

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


            cbModel.SelectedIndex = cbModel.Items.Count - 1;
            cbVAE.SelectedIndex = 0;
        }

        void ClearImages()
        {
            Helper.ImgList.Clear();
            ViewImg.Source = NoImageData;

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
            if (Text == null)
            {
                return Text;
            }

            string NewPrompt = Text.Replace(' ', '_');

            NewPrompt = NewPrompt.Replace(")_", ") ");
            NewPrompt = NewPrompt.Replace("(_", "(");

            NewPrompt = NewPrompt.Replace("_)", ")");
            NewPrompt = NewPrompt.Replace("_(", " (");
            
            NewPrompt = NewPrompt.Replace("_[", "[");
            NewPrompt = NewPrompt.Replace("_]", " ]");

            NewPrompt = NewPrompt.Replace("]_", "] ");
            NewPrompt = NewPrompt.Replace("[_", "[");

            NewPrompt = NewPrompt.Replace(",_", ", ");
            NewPrompt = NewPrompt.Replace("_,", " ,");

            NewPrompt = NewPrompt.Replace(":_", ": ");
            NewPrompt = NewPrompt.Replace("_:", " :");

            NewPrompt = NewPrompt.Replace("-_", "- ");
            NewPrompt = NewPrompt.Replace("_-", " -");
            NewPrompt = NewPrompt.Replace(" _", string.Empty);

            NewPrompt = NewPrompt.Replace("*", string.Empty);

            if (NewPrompt[NewPrompt.Length - 1] != ',')
                NewPrompt += ",";

            return NewPrompt;
        }
    }
}
