namespace SD_FXUI
{
    internal class Schedulers
    {
        public static string[] Shark =
        {
            "DDIM",
            "EulerDiscrete",
            "EulerAncestralDiscrete",
            "DPMSolverMultistep",
            "PNDM",
            "LMSDiscrete",
            "SharkEulerDiscrete"
        };

        public static string[] Diffusers =
        {
            "DDIM",
            "EulerDiscrete",
            "EulerAncestralDiscrete",
            "DPMSolverMultistep",
            "PNDM",
            "LMSDiscrete",
            "DDPM",

            // Extra steps schedulers 
            "DPMDiscrete",
            "HeunDiscrete"
        };
    }
}
