using HandyControl.Data;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

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

			cbDevice.Text = Data.Get("device");

			bool FirstStart = Data.Get("welcomewnd") != "true";
			if (FirstStart)
			{
				Welcome Hellow = new Welcome();
				Hellow.ShowDialog();

				switch (GlobalVariables.Mode)
				{
					case Helper.ImplementMode.ONNX: GlobalVariables.Mode = Helper.ImplementMode.IDK; btnONNX_Click(0, new RoutedEventArgs()); break;
					case Helper.ImplementMode.DiffCUDA: GlobalVariables.Mode = Helper.ImplementMode.IDK; btnDiffCuda_Click(0, new RoutedEventArgs()); break;

					default: btnONNX_Click(0, new RoutedEventArgs()); break;
				}
			}
			else
			{
				switch (Data.Get("back_mode"))
				{
					case "1": btnONNX_Click(0, new RoutedEventArgs()); break;
					case "2": btnDiffCpu_Click(0, new RoutedEventArgs()); break;
					case "3": GlobalVariables.Mode = Helper.ImplementMode.InvokeAI; break;
					case "0": btnDiffCuda_Click(0, new RoutedEventArgs()); break;

					default: btnONNX_Click(0, new RoutedEventArgs()); break;
				}
			}

			UpdateModelsList();
			cbModel.Text = Data.Get("model");
			cbVAE.Text = Data.Get("VAE");
			cbLoRA.Text = Data.Get("lora");
			cbLoRACat.Text = Data.Get("lora_cat");

			if (cbVAE.Text.Length == 0)
			{
				cbVAE.Text = "Default";
			}

			var HistoryStack = Data.Get("history").Split('|');
			foreach (var item in HistoryStack)
			{
				if (item.Length > 0)
                    GlobalVariables.PromHistory.Add(item);
			}

			// Disable button for user
			btnDiffCuda.IsEnabled = GlobalVariables.GPUID.GreenGPU;
			btnONNX.IsEnabled = GlobalVariables.GPUID.RedGPU || GlobalVariables.GPUID.NormalBlueGPU;
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
			Data.Set("lora_cat", cbLoRACat.Text);
			Data.Set("cfg", tbCFG.Text);

			Data.Set("model", cbModel.Text);
			Data.Set("VAE", cbVAE.Text);
			Data.Set("back_mode", ((int)(GlobalVariables.Mode)).ToString());

			string HistoryStack = "";
			foreach (var item in GlobalVariables.PromHistory)
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
			string Prompt = GlobalVariables.MakeInfo.Prompt;

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

			if (GlobalVariables.DrawMode == Helper.DrawingMode.Img2Img)
			{
				float Denoising = float.Parse(tbDenoising.Text);
				Denoising /= 100;

				string newDenoising = Denoising.ToString().Replace(",", ".");

				if (cbPix2Pix.IsChecked.Value)
				{
					CmdLine += $" --mode=\"pix2pix\" --img=\"{GlobalVariables.InputImagePath}\" --imgscale={newDenoising}";
				}
				else
				{
					if (GlobalVariables.ImgMaskPath != string.Empty)
					{
						CmdLine += $" --mode=\"inpaint\" --img=\"{GlobalVariables.InputImagePath}\" --imgscale={newDenoising}";
						CmdLine += $" --imgmask=\"{GlobalVariables.ImgMaskPath}\"";
					}
					else
					{
						CmdLine += $" --mode=\"img2img\" --img=\"{GlobalVariables.InputImagePath}\" --imgscale={newDenoising}";
					}
				}

				if (!File.Exists(GlobalVariables.InputImagePath))
				{
					Notification.MsgBox("Incorrect image path!");
					CmdLine = "";
				}
			}

			return CmdLine;
		}
		private bool MakeCommandObject()
		{
			GlobalVariables.MakeInfo.LoRA.Clear();
			GlobalVariables.MakeInfo.TI.Clear();
			GlobalVariables.MakeInfo.TINeg.Clear();

			string SourcePrompt = CodeUtils.GetRichText(tbPrompt);
			SourcePrompt = LoRA.Extract(SourcePrompt);
            SourcePrompt = TextualInversion.Extract(SourcePrompt);

            string NegPrompt = CodeUtils.GetRichText(tbNegPrompt);
            NegPrompt = TextualInversion.Extract(NegPrompt, false);

			GlobalVariables.MakeInfo.Prompt = SourcePrompt;
			GlobalVariables.MakeInfo.NegPrompt = NegPrompt;
			GlobalVariables.MakeInfo.StartSeed = long.Parse(tbSeed.Text);
			GlobalVariables.MakeInfo.BatchSize = (int)tbParallel.Value;
			GlobalVariables.MakeInfo.CFG = float.Parse(tbCFG.Text);
			GlobalVariables.MakeInfo.Steps = (int)slSteps.Value;
			GlobalVariables.MakeInfo.Model = cbModel.Text;

			if (cbVAE.Text.ToLower() != "default")
			{
				if (cbVAE.Text.StartsWith("vae\\"))
				{
					GlobalVariables.MakeInfo.VAE = FS.GetModelDir() + cbVAE.Text.ToLower();
				}
				else
				{
					string Dir = (GlobalVariables.Mode == Helper.ImplementMode.ONNX) ? "onnx\\" : "diffusers\\";

					GlobalVariables.MakeInfo.VAE = FS.GetModelDir() + Dir + cbVAE.Text.ToLower();
				}
			}
			else
			{
				GlobalVariables.MakeInfo.VAE = cbVAE.Text;
			}

			GlobalVariables.MakeInfo.Sampler = cbSampler.Text;
			GlobalVariables.MakeInfo.ETA = (int)slETA.Value;
			GlobalVariables.MakeInfo.TotalCount = (int)tbTotalCount.Value;
			GlobalVariables.MakeInfo.Height = (int)slH.Value;
			GlobalVariables.MakeInfo.Width = (int)slW.Value;
			GlobalVariables.MakeInfo.Device = (GlobalVariables.Mode == Helper.ImplementMode.DiffCUDA) ? "cuda" : "cpu";
			GlobalVariables.MakeInfo.ImgScale = 0;
			GlobalVariables.MakeInfo.WorkingDir = FS.GetWorkingDir();

			if (cbPix2Pix.IsChecked.Value && GlobalVariables.DrawMode == Helper.DrawingMode.Img2Img)
            {
                string WorkingDir = FS.GetModelDir() + "pix2pix/";

                if (!Directory.Exists(WorkingDir))
                {
                    Task.Run(() => WGetDownloadModels.DownloadPix2Pix());
                    return false;
                }

                GlobalVariables.MakeInfo.Model = WorkingDir;
                GlobalVariables.MakeInfo.Mode = "pix2pix";
				GlobalVariables.MakeInfo.Image = GlobalVariables.InputImagePath;
                GlobalVariables.MakeInfo.ImgCFGScale = (float)slDenoising.Value / 100;
            }
			else if (GlobalVariables.DrawMode == Helper.DrawingMode.Img2Img)
			{
				GlobalVariables.MakeInfo.Mode = "img2img";
				GlobalVariables.MakeInfo.Image = GlobalVariables.InputImagePath;
				GlobalVariables.MakeInfo.ImgScale = (float)slDenoising.Value / 100;
			}
			else if (GlobalVariables.DrawMode == Helper.DrawingMode.Inpaint)
			{
				GlobalVariables.MakeInfo.Mode = "inpaint";
				GlobalVariables.MakeInfo.Image = GlobalVariables.InputImagePath;
				GlobalVariables.MakeInfo.Mask = GlobalVariables.ImgMaskPath;
				GlobalVariables.MakeInfo.ImgScale = (float)slDenoising.Value / 100;
			}
			else if (GlobalVariables.DrawMode == Helper.DrawingMode.Text2Img)
				GlobalVariables.MakeInfo.Mode = "txt2img";

			int Size = (int)slUpscale.Value;
            GlobalVariables.CurrentUpscaleSize = Size;

			return true;
		}

		void ValidateSize()
		{
			if (GlobalVariables.Mode != Helper.ImplementMode.ONNX) 
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
			string Prompt = GlobalVariables.MakeInfo.Prompt;
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

			if (cbNSFW.IsChecked.Value)
			{
				CmdLine += " --nsfw=True";
			}

			if (CPUUse)
			{
				CmdLine += " --device=\"cpu\"";
			}

			if (GlobalVariables.DrawMode == Helper.DrawingMode.Img2Img)
			{
				float Denoising = float.Parse(tbDenoising.Text);
				Denoising /= 100;

				string newDenoising = Denoising.ToString().Replace(",", ".");

				if (cbPix2Pix.IsChecked.Value)
				{
					CmdLine += $" --mode=\"pix2pix\" --img=\"{GlobalVariables.InputImagePath}\" --imgscale={newDenoising}";
				}
				else
				{
					if (GlobalVariables.ImgMaskPath != string.Empty)
					{
						CmdLine += $" --mode=\"inpaint\" --img=\"{GlobalVariables.InputImagePath}\" --imgscale={newDenoising}";
						CmdLine += $" --imgmask=\"{GlobalVariables.ImgMaskPath}\"";
					}
					else
					{
						CmdLine += $" --mode=\"img2img\" --img=\"{GlobalVariables.InputImagePath}\" --imgscale={newDenoising}";
					}
				}

				if (!File.Exists(GlobalVariables.InputImagePath))
				{
					Notification.MsgBox("Incorrect image path!");
					CmdLine = "";
				}
			}

			return CmdLine;
		}

		void SetImg(string Img, bool CN = false)
		{
			ViewImg.Source = CodeUtils.BitmapFromUri(new Uri(Img));
			ListViewItemsCollections.Add(new ListViewItemsData()
			{
				GridViewColumnName_ImageSource = Img,
				GridViewColumnName_LabelContent = "Seed: " + (GlobalVariables.MakeInfo.StartSeed + ListViewItemsCollections.Count).ToString()
			});

			lvImages.ItemsSource = ListViewItemsCollections;
            GlobalVariables.ImgList.Add(Img);

			btnDDB.Visibility = Visibility.Visible;

			if (CN)
			{
				imgPose.Source = CodeUtils.BitmapFromUri(new Uri(Img));
                GlobalVariables.CurrentPose = Img;

				UpdateModelsListControlNet();
				cbPose.SelectedIndex = cbPose.Items.Count - 1;
			}
		}

		void UpdateLoRAModels(string LoraPath)
		{
			LoRA.Reload();

            foreach (var Itm in Directory.GetFiles(LoraPath))
			{
				string TryName = Path.GetFileNameWithoutExtension(Itm);

				if (!Itm.EndsWith("txt"))
					cbLoRA.Items.Add(TryName);
			}
		}

		public void UpdateModelsList()
		{
			string SafeVAE = (string)cbVAE.Text;
			string SafeModelName = (string)cbModel.Text;
			string SafeLoRAName = (string)cbLoRACat.Text;

			cbModel.Items.Clear();
			cbVAE.Items.Clear();
			cbTI.Items.Clear();
			cbLoRACat.Items.Clear();

			cbVAE.Items.Add("Default");
			foreach (var Itm in FS.GetModels(GlobalVariables.Mode))
			{
				cbModel.Items.Add(Itm);

				if (!Itm.EndsWith("hgf") && Utils.Settings.UseInternalVAE)
					cbVAE.Items.Add(Itm);
			}

			// Textual Inversion
			TextualInversion.Reload();
			foreach (string Dir in Directory.GetFiles(FS.GetModelDir(FS.ModelDirs.TextualInversion)))
            {
                cbTI.Items.Add(Path.GetFileNameWithoutExtension(Dir));
            }

			// Yeah... LoRA...
			string LoraPath = FS.GetModelDir(FS.ModelDirs.LoRA);

			foreach (string Dir in Directory.GetDirectories(LoraPath))
			{
				cbLoRACat.Items.Add(Path.GetFileName(Dir));
			}

			int NoneID = cbLoRACat.Items.Add("None");
			cbLoRACat.SelectedIndex = NoneID;

			// VAE
			foreach (var Itm in Directory.GetDirectories(FS.GetModelDir() + "vae\\"))
			{
				if (GlobalVariables.Mode == Helper.ImplementMode.ONNX)
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

			if (SafeLoRAName.Length> 0)
			{
				cbLoRACat.Text = SafeLoRAName;
			}
		}

		void ClearImages()
		{
            GlobalVariables.ImgList.Clear();
			ViewImg.Source = GlobalVariables.NoImageData;

			lvImages.UnselectAll();
			ListViewItemsCollections.Clear();

            GlobalVariables.ActiveImageState = Helper.ImageState.Free;
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
				imgPose.Source = GlobalVariables.NoImageData;
			}

			cbPreprocess.Items.Add(HelperControlNet.Current.GetModelName());
			cbPreprocess.SelectedIndex = 0;
		}
	}
}