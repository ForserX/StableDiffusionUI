using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media.Imaging;
using System.Windows.Media;

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
            return new TextRange(Tb.Document.ContentStart, Tb.Document.ContentEnd).Text;
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

    }
}
