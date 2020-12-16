using Ebby.Gym.Envs.Classic;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Pendulum.Net
{
    public static class CartPole
    {
        private static void CartPoleEnv()
        {
            CartPoleEnv cp = new CartPoleEnv(); //or AvaloniaEnvViewer.Factory
            bool done = true;
            for (int i = 0; i < 100_000; i++)
            {
                if (done)
                {
                    NDArray observation = cp.Reset();
                    done = false;
                }
                else
                {
                    var (observation, reward, _done, information) = cp.Step((i % 2)); //we switch between moving left and right
                    done = _done;

                    if (done)
                    {

                    }
                    //do something with the reward and observation.
                }

                //var view = new Ebby.Gym.Rendering.Viewer(100, 100, "viewer");
                var img = cp.Render(); //returns the image that was rendered.

                Thread.Sleep(10); //this is to prevent it from finishing instantly !
            }

        }
    }
}
