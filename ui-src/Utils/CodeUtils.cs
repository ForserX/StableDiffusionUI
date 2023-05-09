using System;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Text.RegularExpressions;

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
