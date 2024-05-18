using System.Runtime.InteropServices;

namespace ML
{
    public class NNModel
    {
        [DllImport("NeuralNetwork.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Add(int a, int b);
    }
}
