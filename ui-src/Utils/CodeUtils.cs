﻿using System;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Text.RegularExpressions;
using System.Drawing.Imaging;
using System.Drawing;
using System.Windows;

namespace SD_FXUI
{
	internal class CodeUtils
	{
		public static void SetRichText(RichTextBox Tb, string Text)
		{
			Tb.Document.Blocks.Clear();
			Tb.Document.Blocks.Add(new Paragraph(new Run(Text)));
		}

		public static string GetRichText(RichTextBox Tb)
		{
			string ResultString = new TextRange(Tb.Document.ContentStart, Tb.Document.ContentEnd).Text;
			ResultString = ResultString.Replace("\r\n", string.Empty);

			return ResultString;
		}
		
		public static ImageSource BitmapFromUri(System.Uri source)
		{
			var bitmap = new BitmapImage();
			bitmap.BeginInit();
			bitmap.UriSource = source;
			bitmap.CacheOption = BitmapCacheOption.OnLoad;
			bitmap.EndInit();
			return bitmap;
		}

        public static Bitmap BitmapFromSource(ImageSource source)
        {
            BitmapSource NormalBitmap = source as BitmapSource;
            
            Bitmap bmp = new Bitmap
			(
                NormalBitmap.PixelWidth,
                NormalBitmap.PixelHeight,
				System.Drawing.Imaging.PixelFormat.Format32bppPArgb
			);

			BitmapData data = bmp.LockBits
			(
				new Rectangle
				(
				  System.Drawing.Point.Empty,
				  bmp.Size
				),
				ImageLockMode.WriteOnly,
				System.Drawing.Imaging.PixelFormat.Format32bppPArgb
			 );

            NormalBitmap.CopyPixels
			(
				Int32Rect.Empty,
				data.Scan0,
				data.Height * data.Stride,
				data.Stride
			);

			bmp.UnlockBits(data);

			return bmp;
		}

        public static BitmapSource CreateBitmapSourceFromGdiBitmap(Bitmap bitmap)
        {
            if (bitmap == null)
                throw new ArgumentNullException("bitmap");

            var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);

            var bitmapData = bitmap.LockBits(
                rect,
                ImageLockMode.ReadWrite,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            try
            {
                var size = (rect.Width * rect.Height) * 4;

                return BitmapSource.Create(
                    bitmap.Width,
                    bitmap.Height,
                    bitmap.HorizontalResolution,
                    bitmap.VerticalResolution,
                    PixelFormats.Bgra32,
                    null,
                    bitmapData.Scan0,
                    size,
                    bitmapData.Stride);
            }
            finally
            {
                bitmap.UnlockBits(bitmapData);
            }
        }
        public static Bitmap ResizeBitmap(Bitmap imgToResize, System.Drawing.Size size)
        {
            try
            {
                Bitmap b = new Bitmap(size.Width, size.Height);
                using (Graphics g = Graphics.FromImage((System.Drawing.Image)b))
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.DrawImage(imgToResize, 0, 0, size.Width, size.Height);
                }

                return b;
            }
            catch
            {
                Host.Print("Bitmap could not be resized");
                return imgToResize;
            }
        }

        public static string MetaData(string File)
		{
			string MetaText = "No meta";
			var Data = MetadataExtractor.ImageMetadataReader.ReadMetadata(File);
			if (Data[1].Tags[0].Description != null)
			{
				MetaText = Data[1].Tags[0].Description;
				MetaText = MetaText.Replace("XUI Metadata: ", string.Empty);
			}

			return MetaText;
		}

		public static string Data()
		{
			string OutData = "";

			DateTime time = DateTime.Now;

			if (time.Day < 10)
				OutData += "0";
			OutData += time.Day + ".";
			if (time.Month < 10)
				OutData += "0";
			OutData += time.Month + "." + time.Year + "_";
			if (time.Hour < 10)
				OutData += "0";
			OutData += time.Hour + "-";
			if (time.Minute < 10)
				OutData += "0";
			OutData += time.Minute + "-";
			if (time.Second < 10)
				OutData += "0";
			OutData += time.Second;

			return OutData;
		}

		public static string ToUpperFirstLetter(string source)
		{
			if (string.IsNullOrEmpty(source))
				return string.Empty;

			// convert to char array of the string
			char[] letters = source.ToCharArray();

			// upper case the first char
			letters[0] = char.ToUpper(letters[0]);

			// return the array made of the new char array
			return new string(letters);
		}
	}
}
