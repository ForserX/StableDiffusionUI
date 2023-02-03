using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class Config
    {
        Configuration ConfigFile = null;
        public Config()
        {
            ConfigFile = ConfigurationManager.OpenExeConfiguration(ConfigurationUserLevel.None);
        }

        public string Get(string Key)
        {
            if (ConfigFile.AppSettings.Settings[Key] != null)
                return ConfigFile.AppSettings.Settings[Key].Value;

            return "";
        }

        public void Set(string Key, string Value)
        {
            if (ConfigFile.AppSettings.Settings[Key] != null)
                ConfigFile.AppSettings.Settings[Key].Value = Value;
            else
                ConfigFile.AppSettings.Settings.Add(Key, Value);
        }

        public void Save()
        {
            ConfigFile.Save(ConfigurationSaveMode.Full, true);
            ConfigurationManager.RefreshSection("appSettings");
        }
    }
}
