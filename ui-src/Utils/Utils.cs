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
    }
}
