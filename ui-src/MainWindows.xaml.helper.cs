using HandyControl.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace SD_FXUI
{
    public partial class MainWindow
    {
        public void SetPrompt(string Prompt) => TryPrompt.Text = Prompt;
        public void UpdateViewImg(string Img) => Dispatcher.Invoke(() => SetImg(Img));
        public void InvokeClearImages() => Dispatcher.Invoke(() => { InvokeClearImages(); });
        public void InvokeProgressUpdate(int value) => Dispatcher.Invoke(() => { pbGen.Value = value; });
        public void InvokeProgressApply() => Dispatcher.Invoke(() => { pbGen.Value += (40 / int.Parse(tbTotalCount.Text)); });

        void Load()
        {
            cbFf16.IsChecked = Data.Get("fp16") == "true";
            cbGfpgan.IsChecked = Data.Get("cbGfpgan") == "true";
            cbY.Text = Data.Get("height");
            cbX.Text = Data.Get("width");
            NegPrompt.Text = Data.Get("neg");
            tbSteps.Text = Data.Get("steps");
            cbUpscaler.Text = Data.Get("upscaler");
            cbSampler.Text = Data.Get("sampler");
            var ListModel = Data.Get("model");

            switch (Data.Get("back_mode"))
            {
                case "0": Helper.Mode = Helper.ImplementMode.InvokeAI; break;
                case "1": btnShark_Click(0, new RoutedEventArgs()); break;
                case "2": btnONNX_Click(0, new RoutedEventArgs()); break;
                case "4": btnDiffCpu_Click(0, new RoutedEventArgs()); break;
                case "3": btnDiffCuda_Click(0, new RoutedEventArgs()); break;

                default: btnONNX_Click(0, new RoutedEventArgs()); break;
            }

            UpdateModelsList();
            cbModel.Text = ListModel.ToString();
            cbDevice.Text = Data.Get("device");
            cbVAE.Text = Data.Get("VAE");

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
            Data.Set("cbGfpgan", cbGfpgan.IsChecked == true ? "true" : "false");
            Data.Set("height", cbY.Text);
            Data.Set("width", cbX.Text);
            Data.Set("neg", NegPrompt.Text);
            Data.Set("steps", tbSteps.Text);
            Data.Set("upscaler", cbUpscaler.Text);
            Data.Set("sampler", cbSampler.Text);
            Data.Set("device", cbDevice.Text);

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

            string CmdLine = $""
                    + $" --prompt=\"{TryPrompt.Text}\""
                    + $" --prompt_neg=\"{NegPrompt.Text}\""
                    + $" --height={cbY.Text}"
                    + $" --width={cbX.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --totalcount={tbTotalCount.Text}"
                    + $" --model=\"{Model}\""
                    + $" --vae=\"{VAE}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale=0.{tbDenoising.Text}";
            }

            return CmdLine;
        }
        private string GetCommandLineDiffCuda()
        {
            string Model = FS.GetModelDir() + "diff\\" + cbModel.Text;
            string VAE = cbVAE.Text.ToLower();
            if (VAE != "default")
            {
                if (VAE.StartsWith("vae\\"))
                {
                    VAE = FS.GetModelDir() + cbVAE.Text.ToLower();
                }
                else
                {
                    VAE = FS.GetModelDir() + "diff\\" + cbVAE.Text.ToLower();
                }
            }

            string CmdLine = $""
                    + $" --prompt=\"{TryPrompt.Text}\""
                    + $" --prompt_neg=\"{NegPrompt.Text}\""
                    + $" --height={cbY.Text}"
                    + $" --width={cbX.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --totalcount={tbTotalCount.Text}"
                    + $" --model=\"{Model}\""
                    + $" --vae=\"{VAE}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if(CPUUse)
            {
                CmdLine += " --device=\"cpu\"";
            }

            if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale=0.{tbDenoising.Text}";
            }

            return CmdLine;
        }
        private string GetCommandLineShark()
        {
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string Model = cbModel.Text.IndexOf('/') != -1 ? cbModel.Text : FS.GetModelDir() + "diff\\" + cbModel.Text;
            string CmdLine = $" --precision={FpMode}"
                    + $" --device=\"{cbDevice.Text}\""
                    + $" --prompt=\"{TryPrompt.Text}\""
                    + $" --negative-prompts=\"{NegPrompt.Text}\""
                    + $" --height={cbY.Text}"
                    + $" --width={cbX.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scheduler={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --total_count={tbTotalCount.Text}"
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

            ListView1.ItemsSource = ListViewItemsCollections;
            ListView1.SelectedIndex = 0;
            ImgList.Add(Img);
        }

        public void UpdateModelsList()
        {
            cbModel.Items.Clear();
            cbVAE.Items.Clear();

            cbVAE.Items.Add("Default");
            foreach (var Itm in FS.GetModels(Helper.Mode))
            {
                cbModel.Items.Add(Itm);
                cbVAE.Items.Add(Itm);
            }
            foreach (var Itm in System.IO.Directory.GetDirectories(FS.GetModelDir()+"vae\\"))
            {
                
                cbVAE.Items.Add("vae\\"+System.IO.Path.GetFileName(Itm));
            }


            cbModel.SelectedIndex = cbModel.Items.Count - 1;
            cbVAE.SelectedIndex = 0;
        }

        void ClearImages()
        {
            ImgList.Clear();
            ViewImg.Source = NoImageData;

            ListView1.UnselectAll();
            ListViewItemsCollections.Clear();

            Helper.ActiveImageState = Helper.ImageState.Free;
            btnFavor.Source = imgNotFavor.Source;
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
    }
}
