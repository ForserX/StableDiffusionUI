﻿using System.Collections;
using System.Threading.Tasks;
using System.Windows;

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для SharkModelImporter.xaml
    /// </summary>
    public partial class SharkModelImporter : Window
    {
        string SwapText = "Enter the path to the file (.ckpt/.safetensors/.pt) or folder (diffusers)";

        public SharkModelImporter()
        {
            InitializeComponent();

            cbFrom.SelectedIndex = 0;
            cbTo.SelectedIndex = 0;
            cbBaseModel.SelectedIndex = 0;
            cbPath.Text = SwapText;
        }
        
        bool PresetIs768()
        {
            return cbBaseModel.Text.Contains("768");
        }

        string PresetName()
        {
            if (cbBaseModel.Text.Contains("768"))
            {
                return "v2-inference-v.yaml";
            }
            else if (cbBaseModel.Text.Contains("clip"))
            {
                return "v2-1-stable-unclip-l-inference.yaml";
            }
            else if (cbBaseModel.Text.Contains("2."))
            {
                return "v2-inference.yaml";
            }
            else
            {
                return "v1-inference.yaml";
            }
        }

        private void OrigToDiff(bool bWait = false)
        {
            if (!System.IO.File.Exists(cbPath.Text))
                return;

            string SafeName = cbPath.Text;

            if(bWait)
            {
                CMD.ProcessConvertCKPT2Diff(SafeName, chBoxEmaOnly.IsChecked.Value, PresetIs768(), PresetName());
            }
            else
            {
                bool EMA = chBoxEmaOnly.IsChecked.Value, 
                     b768 = PresetIs768();

                Task.Run(() => CMD.ProcessConvertCKPT2Diff(SafeName, EMA, b768));
            }
        }

        async Task CastCp2ONNX(string XName)
        {
            string Name = System.IO.Path.GetFileName(XName);
            if (Name.EndsWith(".ckpt") || Name.EndsWith(".safetensors"))
            {
                Name = System.IO.Path.GetFileNameWithoutExtension(XName);
            }

            string SafeName = FS.GetModelDir() + "diffusers\\" + Name;
            Task.Run(() => CMD.ProcessConvertDiff2Onnx(SafeName));
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            int GetFromID = cbFrom.SelectedIndex; 
            int ToID = cbTo.SelectedIndex;

            // Anyway need cast to diff
            if (GetFromID == 0 || GetFromID == 1)
            {
                if(ToID == 0)
                {
                    CMD.ProcessConvertCKPT2ONNX(cbPath.Text, chBoxEmaOnly.IsChecked.Value, PresetIs768(), PresetName());
                }
                else
                {
                    OrigToDiff(ToID == 1);
                }
            }

            if(GetFromID == 2)
            {
                string sCommand = cbPath.Text;

                if (ToID == 0)
                {
                    Task.Run(() => CMD.ProcessConvertVaePt2ONNX(sCommand));
                }
                else
                {
                    Task.Run(() => CMD.ProcessConvertVaePt2Diff(sCommand));
                }    
            }

            if (ToID == 0)
            {
                if(GetFromID == 3)
                {
                    string SafeNameStr = cbPath.Text;
                    await Task.Run(() => CastCp2ONNX(SafeNameStr));
                }
            }
            else
            {
                /*
                if (GetFromID == 4)
                {
                    HuggCast();
                }
                */
            }

            GlobalVariables.Form.UpdateModelsList();
        }

        private void cbTo_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
        }

        private void cbPath_Drop(object sender, DragEventArgs e)
        {
           
            // Note that you can have more than one file.
            cbPath.Text = ((string[]) e.Data.GetData(DataFormats.FileDrop))[0];

            // Assuming you have one file that you care about, pass it off to whatever
            // handling code you have defined.

            if (GlobalVariables.Mode == Helper.ImplementMode.ONNX)
            {
                cbTo.SelectedIndex = 0;
            }
            else
            {
                cbTo.SelectedIndex = 1;
            }

            if (!FS.HasExt(cbPath.Text, new string[] { ".pt", ".ckpt", ".safetensors" }) && !FS.IsDirectory(cbPath.Text)) 
            {
                cbPath.Text = "";
                Notification.MsgBox("Incorrect file!");
            }

            if (cbPath.Text.EndsWith("vae-ft-mse-840000-ema-pruned.ckpt") || cbPath.Text.EndsWith("kl-f8-anime2.ckpt"))
            {
                cbFrom.SelectedIndex = 2;  // this vae .pt file In fact
            }
            else if (cbPath.Text.EndsWith(".ckpt")) cbFrom.SelectedIndex = 0;
            else if (cbPath.Text.EndsWith(".safetensors")) cbFrom.SelectedIndex = 1;
            else if (cbPath.Text.EndsWith(".pt"))
            {
                cbFrom.SelectedIndex = 2;
            }
            else cbFrom.SelectedIndex = 3;
        }

        private void cbPath_DragEnter(object sender, DragEventArgs e)
        {           
            cbPath.Visibility = Visibility.Collapsed;
        }

        private void cbPath_DragLeave(object sender, DragEventArgs e)
        {
            cbPath.Visibility = Visibility.Visible;
        }

        private void Grid_Drop(object sender, DragEventArgs e)
        {
            if (null != e.Data && e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                //var data = e.Data.GetData(DataFormats.FileDrop) as string[];
                // handle the files here!
                cbPath.Opacity = 1.0;
                cbPath_Drop(sender, e);
            }

            cbPath.Visibility = Visibility.Visible;
        }


        private void Grid_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy;
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }
        }

        private void cbPathLost(object sender, RoutedEventArgs e)
        {
            if (cbPath.Text == "")
                cbPath.Text = SwapText;
        }

        private void cbPathSet(object sender, RoutedEventArgs e)
        {
            if (cbPath.Text == SwapText)
                cbPath.Text = "";
        }
    }
}
