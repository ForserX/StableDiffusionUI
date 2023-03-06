using Microsoft.Toolkit.Uwp.Notifications;
using Microsoft.Win32;
using SD_FXUI.Utils.Models;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace SD_FXUI
{
    public partial class MainWindow : HandyControl.Controls.BlurWindow
    {
        Config Data = null;
        ImageSource NoImageData = null;
        ObservableCollection<ListViewItemsData> ListViewItemsCollections = new ObservableCollection<ListViewItemsData>();
        string currentImage = null;

        public class ListViewItemsData
        {
            public string? GridViewColumnName_ImageSource { get; set; }
            public string? GridViewColumnName_ID { get; set; }
        }

        bool CPUUse = false;
        public MainWindow()
        {
            InitializeComponent();

            Install.SetupDirs();

            cbUpscaler.SelectedIndex = 0;
            cbModel.SelectedIndex = 0;

            cbSampler.SelectedIndex = 0;
            cbDevice.SelectedIndex = 0;

            Helper.Form = this;

            Helper.UIHost = new HostForm();
            Helper.UIHost.Hide();
            Host.Print("\n");

            Helper.GPUID = new GPUInfo();

            // Load App data
            Data = new Config();
            Load();
            ChangeTheme();

            gridImg.Visibility = Visibility.Collapsed;
            btnDDB.Visibility = Visibility.Collapsed;
            NoImageData = ViewImg.Source;
            Helper.SafeMaskFreeImg = imgMask.Source;

            ToastNotificationManagerCompat.OnActivated += toastArgs =>
            {
                Notification.ToastBtnClickManager(toastArgs);
            };

            cbTI.IsEnabled = false;
            btnApplyTI.IsEnabled = false;
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            Directory.CreateDirectory(Helper.ImgPath);
            btnDDB.Visibility = Visibility.Collapsed;

            if (chRandom.IsChecked.Value)
            {
                var rand = new Random();
                tbSeed.Text = rand.Next().ToString();
            }

            int Size = (int)slUpscale.Value;

            string cmdline = "";
            bool SafeCPUFlag = CPUUse;

#pragma warning disable CS4014
            switch (Helper.Mode)
            {
                case Helper.ImplementMode.Shark:
                    {
                        cmdline += GetCommandLineShark();
                        Task.Run(() => CMD.ProcessRunnerShark(cmdline, Size));
                        break;
                    }
                case Helper.ImplementMode.ONNX:
                    {
                        cmdline += GetCommandLineOnnx();
                        Task.Run(() => CMD.ProcessRunnerOnnx(cmdline, Size));
                        break;
                    }
                case Helper.ImplementMode.DiffCPU:
                case Helper.ImplementMode.DiffCUDA:
                    {
                        cmdline += GetCommandLineDiffCuda();
                        Task.Run(() => CMD.ProcessRunnerDiffCuda(cmdline, Size, SafeCPUFlag));
                        break;
                    }
            }

#pragma warning restore CS4014
            if (Helper.PromHistory.Count == 0 || Helper.PromHistory[0] != TryPrompt.Text)
            {
                Helper.PromHistory.Insert(0, TryPrompt.Text);
            }

            currentImage = null;
            ClearImages();
            InvokeProgressUpdate(3);
        }

        private void Slider_Denoising(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbDenoising != null)
                tbDenoising.Text = slDenoising.Value.ToString();
        }
        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbSteps != null)
                tbSteps.Text = slSteps.Value.ToString();
        }

        private void tbSteps2_TextChanged(object sender, TextChangedEventArgs e)
        {
            double Val = 0;
            double.TryParse(tbCFG.Text, out Val);
            slCFG.Value = Val;
        }

        private void Slider2_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbCFG != null)
                tbCFG.Text = slCFG.Value.ToString();
        }

        private void slUpscale_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (lbUpscale != null)
                lbUpscale.Content = "x" + (slUpscale.Value + 1).ToString();
        }

        private void tbSteps_TextChanged(object sender, TextChangedEventArgs e)
        {
            double Val = 0;
            double.TryParse(tbSteps.Text, out Val);
            slSteps.Value = Val;
        }

        private void btFolder_ValueChanged(object sender, MouseButtonEventArgs e)
        {
            string argument = "/select, \"" + Helper.ImgPath + "\"";
            Host Explorer = new Host("", "explorer.exe");
            Explorer.Start(argument);
        }
        private void btCmd_ValueChanged(object sender, MouseButtonEventArgs e)
        {
            Helper.UIHost.Hide();
            Helper.UIHost.Show();
        }

        private void OnClose(object sender, EventArgs e)
        {
            Helper.UIHost.Close();
            Save();
        }

        private void Button_ClickBreak(object sender, RoutedEventArgs e)
        {
            foreach (var Proc in Helper.SecondaryProcessList)
            {
                Proc.Kill();
            }

            Host.Print("\n All task aborted (」°ロ°)」");
            Helper.SecondaryProcessList.Clear();
            InvokeProgressUpdate(0);
        }

        private void Button_Click_Import_Model(object sender, RoutedEventArgs e)
        {
            Utils.SharkModelImporter Importer = new Utils.SharkModelImporter();
            Importer.Show();
        }

        private void chRandom_Checked(object sender, RoutedEventArgs e)
        {
            if (tbSeed != null)
                tbSeed.IsEnabled = false;
        }
        private void chRandom_Unchecked(object sender, RoutedEventArgs e)
        {
            tbSeed.IsEnabled = true;
        }

        private void cbDevice_TextChanged(object sender, RoutedEventArgs e)
        {

        }

        private void btnONNX_Click(object sender, RoutedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.ONNX)
            {
                Helper.Mode = Helper.ImplementMode.ONNX;
                Install.CheckAndInstallONNX();

                var Safe = btnONNX.Background;
                btnONNX.Background = new SolidColorBrush(Colors.DarkOrchid);
                btnShark.Background = Safe;
                btnDiffCuda.Background = Safe;
                btnDiffCpu.Background = Safe;

                UpdateModelsList();
                UpdateModelsTIList();

                cbDevice.Items.Clear();

                foreach (var item in Helper.GPUID.GPUs)
                {
                    cbDevice.Items.Add(item);
                }

                btnImg.Visibility = Visibility.Visible;
                cbFf16.Visibility = Visibility.Hidden;
                grLoRA.Visibility = Visibility.Collapsed;
                grDevice.Visibility = Visibility.Visible;
                grVAE.Visibility = Visibility.Visible;

                cbSampler.Items.Clear();
                foreach (string Name in Schedulers.Diffusers)
                {
                    cbSampler.Items.Add(Name);
                }

                cbSampler.Text = Data.Get("sampler");
                cbDevice.Text = Data.Get("device");

                Title = "Stable Diffusion XUI : ONNX venv";
            }
        }
        private void btnDiffCuda_Click(object sender, RoutedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.DiffCUDA)
            {
                Helper.Mode = Helper.ImplementMode.DiffCUDA;

                Install.CheckAndInstallCUDA();

                var Safe = btnDiffCuda.Background;
                btnDiffCuda.Background = new SolidColorBrush(Colors.DarkCyan);
                btnONNX.Background = Safe;
                btnShark.Background = Safe;
                btnDiffCpu.Background = Safe;

                UpdateModelsList();
                UpdateModelsTIList();

                grDevice.Visibility = Visibility.Collapsed;
                grVAE.Visibility = Visibility.Visible;
                grLoRA.Visibility = Visibility.Visible;

                btnImg.Visibility = Visibility.Visible;
                cbFf16.Visibility = Visibility.Visible;
                CPUUse = false;

                cbSampler.Items.Clear();
                foreach (string Name in Schedulers.Diffusers)
                {
                    cbSampler.Items.Add(Name);
                }

                cbSampler.Text = Data.Get("sampler", "DDIM");

                foreach (var item in Helper.GPUID.GPUs)
                {
                    if (item.Contains("nvidia"))
                    {
                        cbDevice.Items.Add(item);
                    }
                }

                if (cbDevice.Items.Count == 0)
                {
                    cbDevice.Items.Add("None");
                }

                cbDevice.Text = Data.Get("device");

                Title = "Stable Diffusion XUI : CUDA venv";
            }
        }

        private void btnShark_Click(object sender, RoutedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.Shark)
            {
                Helper.Mode = Helper.ImplementMode.Shark;
                Install.CheckAndInstallShark();

                var Safe = btnShark.Background;
                btnShark.Background = new SolidColorBrush(Colors.DarkSlateBlue);
                btnONNX.Background = Safe;
                btnDiffCuda.Background = Safe;
                btnDiffCpu.Background = Safe;

                UpdateModelsList();
                UpdateModelsTIList();

                cbDevice.Items.Clear();
                cbDevice.Items.Add("vulkan");
                cbDevice.Items.Add("CUDA");

                btnImg.Visibility = Visibility.Hidden;
                cbFf16.Visibility = Visibility.Visible;
                grDevice.Visibility = Visibility.Visible;
                grVAE.Visibility = Visibility.Collapsed;
                grLoRA.Visibility = Visibility.Collapsed;

                cbSampler.Items.Clear();
                foreach (string Name in Schedulers.Shark)
                {
                    cbSampler.Items.Add(Name);
                }

                cbSampler.Text = Data.Get("sampler", "DDIM");
                cbDevice.Text = Data.Get("device");

                Title = "Stable Diffusion XUI : Shark venv";
            }
        }
        private void btnDiffCpu_Click(object sender, RoutedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.DiffCPU)
            {
                Helper.Mode = Helper.ImplementMode.DiffCPU;
                Install.CheckAndInstallONNX();

                var Safe = btnDiffCpu.Background;
                btnDiffCpu.Background = new SolidColorBrush(Colors.DarkSalmon);
                btnONNX.Background = Safe;
                btnShark.Background = Safe;
                btnDiffCuda.Background = Safe;

                UpdateModelsList();
                UpdateModelsTIList();

                grDevice.Visibility = Visibility.Collapsed;

                btnImg.Visibility = Visibility.Visible;
                cbFf16.Visibility = Visibility.Visible;
                grVAE.Visibility = Visibility.Visible;
                grLoRA.Visibility = Visibility.Visible;
                CPUUse = true;

                cbSampler.Items.Clear();
                foreach (string Name in Schedulers.Diffusers)
                {
                    cbSampler.Items.Add(Name);
                }

                cbSampler.Text = Data.Get("sampler", "DDIM");
                cbDevice.Text = Data.Get("device");

                Title = "Stable Diffusion XUI : CPU venv";
            }
        }
        private void lvImages_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (Helper.ImgList.Count > 0)
            {
                currentImage = (Helper.ImgList[lvImages.SelectedIndex]);

                ViewImg.Source = FS.BitmapFromUri(new Uri(currentImage));
                string NewCurrentImage = currentImage.Replace("_upscale.", ".");

                if (File.Exists(NewCurrentImage))
                {
                    currentImage = NewCurrentImage;
                }

                string Name = FS.GetImagesDir() + "best\\" + Path.GetFileName(Helper.ImgList[lvImages.SelectedIndex]);

                if (File.Exists(Name))
                {
                    Helper.ActiveImageState = Helper.ImageState.Favor;
                    btnFavor.Source = imgFavor.Source;
                }
                else
                {
                    Helper.ActiveImageState = Helper.ImageState.Free;
                    btnFavor.Source = imgNotFavor.Source;
                }
            }
        }
        private void cbDevice_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (cbDevice.SelectedItem == null)
                return;

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                Install.WrapONNXGPU(cbDevice.SelectedIndex > 0);
            }
        }

        private void btnImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog OpenDlg = new OpenFileDialog();
            OpenDlg.Filter = "Image Files|*.jpg;*.jpeg;*.png| PNG (*.png)|*.png|JPG (*.jpg)|*.jpg|All files (*.*)|*.*";
            OpenDlg.Multiselect = false;

            bool? IsOpened = OpenDlg.ShowDialog();
            if (IsOpened.Value)
            {
                Helper.InputImagePath = OpenDlg.FileName;
                gridImg.Visibility = Visibility.Visible;
                imgLoaded.Source = FS.BitmapFromUri(new Uri(Helper.InputImagePath));
                Helper.DrawMode = Helper.DrawingMode.Img2Img;
            }
        }

        private void btnZoom_Click(object sender, MouseButtonEventArgs e)
        {
            if (currentImage == null || currentImage.Length < 5)
                return;

            Utils.ImageView ImgViewWnd = new Utils.ImageView();
            ImgViewWnd.SetImage(currentImage);
            ImgViewWnd.Show();
        }

        private void btnToImg_Click(object sender, MouseButtonEventArgs e)
        {
            if (currentImage == null && Helper.ImgList.Count <= 0)
            {
                return;
            }
            else if (currentImage != null)
            {
                Helper.InputImagePath = currentImage;
                imgLoaded.Source = FS.BitmapFromUri(new Uri(currentImage));
            }
            else
            {
                int Idx = lvImages.SelectedIndex;
                if (lvImages.SelectedIndex == -1)
                {
                    Idx = lvImages.Items.Count - 1;
                }

                Helper.InputImagePath = Helper.ImgList[Idx];
                imgLoaded.Source = FS.BitmapFromUri(new Uri(Helper.InputImagePath));
            }


            gridImg.Visibility = Visibility.Visible;
            Helper.DrawMode = Helper.DrawingMode.Img2Img;
        }

        private void tbDenoising_TextChanged(object sender, TextChangedEventArgs e)
        {
            slDenoising.Value = float.Parse(tbDenoising.Text.Replace('.', ','));
        }

        private void btnImageClear_Click(object sender, RoutedEventArgs e)
        {
            gridImg.Visibility = Visibility.Collapsed;

            Helper.DrawMode = Helper.DrawingMode.Text2Img;
            imgLoaded.Source = NoImageData;

            // Mask clear
            imgMask.Source = Helper.SafeMaskFreeImg;
            Helper.ImgMaskPath = string.Empty;
            imgMask.Visibility = Visibility.Collapsed;
        }

        private void btnHistory_Click(object sender, MouseButtonEventArgs e)
        {
            Utils.HistoryList HistoryWnd = new Utils.HistoryList();
            HistoryWnd.ShowDialog();
        }

        private void BlurWindow_Loaded(object sender, RoutedEventArgs e)
        {
            Install.Check();
        }

        private void btnFavorClick(object sender, MouseButtonEventArgs e)
        {
            if (lvImages.Items.Count == 0 || lvImages.SelectedItem == null)
            {
                return;
            }

            if (Helper.ImageState.Favor == Helper.ActiveImageState)
            {
                string Name = Path.GetFileName(Helper.ImgList[lvImages.SelectedIndex]);
                File.Delete(FS.GetImagesDir() + "best\\" + Name);
                Helper.ActiveImageState = Helper.ImageState.Free;

                btnFavor.Source = imgNotFavor.Source;
            }
            else
            {
                string Name = Path.GetFileName(Helper.ImgList[lvImages.SelectedIndex]);
                File.Copy(Helper.ImgList[lvImages.SelectedIndex], FS.GetImagesDir() + "best\\" + Name);
                Helper.ActiveImageState = Helper.ImageState.Favor;

                btnFavor.Source = imgFavor.Source;
            }
        }

        private void cbUpscaler_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            Helper.CurrentUpscalerType = (Helper.UpscalerType)cbUpscaler.SelectedIndex;
        }

        private void cbGfpgan_SelectionChanged(object sender, RoutedEventArgs e)
        {
            Helper.EnableGFPGAN = cbGfpgan.IsChecked.Value;
        }

        private void btnDeepDanbooru_Click(object sender, RoutedEventArgs e)
        {
            Task.Run(() => CMD.DeepDanbooruProcess(Helper.InputImagePath));
        }

        private void Button_Click_DeepDanbooru(object sender, RoutedEventArgs e)
        {
            if (currentImage != null && currentImage != "")
            {
                Task.Run(() => CMD.DeepDanbooruProcess(currentImage));
            }
        }

        private void gridDrop(object sender, DragEventArgs e)
        {
            if (null != e.Data && e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                imgView_Drop(sender, e);
            }
        }

        private void imgView_Drop(object sender, DragEventArgs e)
        {
            // Note that you can have more than one file.
            string dropedFile = ((string[])e.Data.GetData(DataFormats.FileDrop))[0];

            if (dropedFile.ToLower().EndsWith(".png") || dropedFile.ToLower().EndsWith(".jpg") || dropedFile.ToLower().EndsWith(".jpeg"))
            {
                currentImage = dropedFile;
                ViewImg.Source = FS.BitmapFromUri(new Uri(dropedFile));
                btnDDB.Visibility = Visibility.Visible;
            }
        }

        private void btnBestOpen_Click(object sender, RoutedEventArgs e)
        {
            string Path = FS.GetImagesDir() + "\\best";

            if (!Directory.Exists(Path))
                return;

            currentImage = null;
            ClearImages();

            var Files = FS.GetFilesFrom(Path, new string[] { "png", "jpg" }, false);
            foreach (string file in Files)
            {
                SetImg(file);
            }
        }

        private void slEMA_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbETA != null)
                tbETA.Text = slETA.Value.ToString();
        }

        private void tbEMA_TextChanged(object sender, TextChangedEventArgs e)
        {
            slETA.Value = float.Parse(tbETA.Text.Replace('.', ','));
        }
        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            Regex regex = new Regex("[^0-9]+");
            e.Handled = regex.IsMatch(e.Text);
        }

        private void FloatNumberValidationTextBox(object sender, System.Windows.Input.TextCompositionEventArgs e)
        {
            if (e.Text.Length > 1)
            {
                float SkipFlt = 0;
                e.Handled = !float.TryParse(e.Text.Replace('.', ','), out SkipFlt);
            }
            else if (e.Text == ",")
            {
                e.Handled = false;
                return;
            }

            Regex regex = new Regex("[^0-9]+");
            e.Handled = regex.IsMatch(e.Text);
        }

        private void btnDownload_Click(object sender, RoutedEventArgs e)
        {
            Utils.HuggDownload DownloadWnd = new Utils.HuggDownload();
            DownloadWnd.Show();
        }

        private void btnImageClearMask_Click(object sender, RoutedEventArgs e)
        {
            btnImageClearMask.Visibility = Visibility.Collapsed;

            imgMask.Source = Helper.SafeMaskFreeImg;
            Helper.ImgMaskPath = string.Empty;
        }

        private void btnSettingsClick(object sender, RoutedEventArgs e)
        {
            Utils.Settings SettingsWnd = new Utils.Settings();
            SettingsWnd.Show();
        }

        private void gridImg_Drop(object sender, DragEventArgs e)
        {
            string dropedFile = ((string[])e.Data.GetData(DataFormats.FileDrop))[0];

            if (dropedFile.ToLower().EndsWith(".png") || dropedFile.ToLower().EndsWith(".jpg") || dropedFile.ToLower().EndsWith(".jpeg"))
            {
                Helper.ImgMaskPath = dropedFile;
                imgMask.Source = FS.BitmapFromUri(new Uri(dropedFile));
                btnImageClearMask.Visibility = Visibility.Visible;
            }
        }

        private void slW_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbW != null)
            {
                tbW.Text = slW.Value.ToString();

                if (lbRatio != null)
                    lbRatio.Content = GetRatio(slW.Value, slH.Value);
            }
        }

        private void slH_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbH != null)
            {
                tbH.Text = slH.Value.ToString();

                if (lbRatio != null)
                    lbRatio.Content = GetRatio(slW.Value, slH.Value);
            }
        }

        private void tbW_TextChanged(object sender, TextChangedEventArgs e)
        {
            slW.Value = float.Parse(tbW.Text.Replace('.', ','));

            if (lbRatio != null)
                lbRatio.Content = GetRatio(slW.Value, slH.Value);
        }

        private void tbH_TextChanged(object sender, TextChangedEventArgs e)
        {
            float NewValue = float.Parse(tbH.Text);
            slH.Value = NewValue;

            if (lbRatio != null)
                lbRatio.Content = GetRatio(slW.Value, slH.Value);
        }

        private void btnApplyTI_Click(object sender, RoutedEventArgs e)
        {
            string Path = FS.GetModelDir() + "diffusers\\" + cbModel.Text;

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                string CPath = FS.GetModelDir() + "onnx\\" + cbModel.Text;

                if (!Directory.Exists(Path))
                {
                    Notification.MsgBox("Error! Need base diffuser model for apply!");
                    return;
                }

                TIApply HelpWnd = new TIApply();
                HelpWnd.ShowDialog();

                Directory.CreateDirectory(CPath + "\\textual_inversion_merges\\");

                if (Helper.CurrentTI != null)
                    Task.Run(() => CMD.ApplyTextInv(Path, CPath, Helper.CurrentTI));
            }
            else
            {
                if (!Directory.Exists(Path))
                {
                    Notification.MsgBox("Error! Need base diffuser model for apply!");
                    return;
                }

                TIApply HelpWnd = new TIApply();
                HelpWnd.ShowDialog();

                Directory.CreateDirectory(Path + "\\textual_inversion_merges\\");

                if (Helper.CurrentTI != null)
                    Task.Run(() => CMD.ApplyTextInvDiff(Path, Helper.CurrentTI));
            }
        }

        private void cbModel_Copy_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {

        }

        private void cbModel_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.ONNX || cbTI == null || e.AddedItems.Count == 0)
                return;

            cbTI.Items.Clear();
            cbTI.Items.Add("None");

            string Mode = "onnx/";

            string ModelPath = FS.GetModelDir() + Mode + e.AddedItems[0] + "/textual_inversion_merges/";

            if (!Directory.Exists(ModelPath))
                return;

            foreach (string File in Directory.GetDirectories(ModelPath))
            {
                cbTI.Items.Add(Path.GetFileNameWithoutExtension(File));
            }
        }

        private void btnMerge_Click(object sender, MouseButtonEventArgs e)
        {
            Utils.Merge MergeWnd = new Utils.Merge();
            MergeWnd.ShowDialog();
        }

        private void pbGen_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (pbGen.Value == 100 || pbGen.Value == 0)
                pbGen.Visibility = Visibility.Collapsed;
            else
                pbGen.Visibility = Visibility.Visible;
        }
    }
}