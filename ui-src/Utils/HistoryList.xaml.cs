using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для HistoryList.xaml
    /// </summary>
    public partial class HistoryList : Window
    {
        public HistoryList()
        {
            InitializeComponent();

            foreach (var Itm in GlobalVariables.PromHistory)
                lbHistory.Items.Add(Itm);
        }

        private void ListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            if (lbHistory == null)
                return;

            GlobalVariables.Form.SetPrompt(lbHistory.SelectedItem.ToString());
            this.Close();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            if (lbHistory == null)
                return;

            if (lbHistory.SelectedItem != null)
            {
                GlobalVariables.PromHistory.Remove(lbHistory.SelectedItem.ToString());
                lbHistory.Items.Remove(lbHistory.SelectedItem);
            }
        }

        public static void ApplyPrompt(string Message)
        {
            if (GlobalVariables.PromHistory.Count == 0 || GlobalVariables.PromHistory[0] != Message)
            {
                if (GlobalVariables.PromHistory.Contains(Message))
                {
                    GlobalVariables.PromHistory.Remove(Message);
                }

                GlobalVariables.PromHistory.Insert(0, Message);
            }

        }

        private new void KeyDown(object sender, KeyEventArgs e)
        {
            if (lbHistory == null)
                return;

            if (lbHistory.SelectedItem == null)
                return;

            if (e.Key == Key.Delete)
            {
                int Idx = lbHistory.SelectedIndex;

                GlobalVariables.PromHistory.Remove(lbHistory.SelectedItem.ToString());
                lbHistory.Items.Remove(lbHistory.SelectedItem);

                lbHistory.SelectedIndex = Idx;
            }
            else if (e.Key == Key.Enter)
            {
                GlobalVariables.Form.SetPrompt(lbHistory.SelectedItem.ToString());
                this.Close();
            }
            else if (e.Key == Key.Escape) 
            {
                this.Close();
            }
        }

        private void lbHistory_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            GlobalVariables.Form.SetPrompt(lbHistory.SelectedItem.ToString());
            this.Close();
        }
    }
}
