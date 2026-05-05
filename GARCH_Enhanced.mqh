//+------------------------------------------------------------------+
//|                    GARCH_Enhanced.mqh v2.05                      |
//|  Advanced GARCH(1,1) + Asymmetry + Multi-TF + Sensitivity       |
//|  Used for: Dynamic Lot Sizing + Confidence Adjustment + Risk    |
//+------------------------------------------------------------------+

// Tuned parameters (persistence ≈ 0.96 — realistic for FX)
input double GARCH_Omega = 0.000007;
input double GARCH_Alpha = 0.08;
input double GARCH_Beta  = 0.88;

// Asymmetry (leverage effect) - optional EGARCH-style
input bool   UseAsymmetry     = true;
input double GARCH_Gamma      = 0.05;     // extra weight on negative returns

// Multi-timeframe GARCH
input ENUM_TIMEFRAMES GARCH_TF1 = PERIOD_M15;
input ENUM_TIMEFRAMES GARCH_TF2 = PERIOD_H1;

//+------------------------------------------------------------------+
//| Advanced GARCH(1,1) with asymmetry and multi-TF                  |
//+------------------------------------------------------------------+
double CalculateAdvancedGARCH(string symbol, int shift = 0)
{
   static double garch[2] = {0.0001, 0.0001};   // [M15, H1]
   static double last_close[2] = {0, 0};

   double prices[3];
   if(CopyClose(symbol, GARCH_TF1, shift, 3, prices) < 3) return garch[0];

   double ret = MathLog(prices[0] / prices[1]);
   double var = GARCH_Omega + GARCH_Alpha * ret * ret + GARCH_Beta * garch[0] * garch[0];

   // Asymmetry (negative returns increase volatility more)
   if(UseAsymmetry && ret < 0)
      var += GARCH_Gamma * ret * ret;

   garch[0] = MathSqrt(MathMax(var, 1e-8));
   last_close[0] = prices[0];

   // Optional H1 GARCH for regime detection
   if(GARCH_TF2 != GARCH_TF1)
   {
      double h1_prices[3];
      if(CopyClose(symbol, GARCH_TF2, shift, 3, h1_prices) >= 3)
      {
         double h1_ret = MathLog(h1_prices[0] / h1_prices[1]);
         double h1_var = GARCH_Omega + GARCH_Alpha * h1_ret * h1_ret + GARCH_Beta * garch[1] * garch[1];
         garch[1] = MathSqrt(MathMax(h1_var, 1e-8));
      }
   }

   // Return weighted average (more weight on current TF)
   return (garch[0] * 0.7 + garch[1] * 0.3);
}

//+------------------------------------------------------------------+
//| GARCH Regime Classification (for dynamic behavior)               |
//+------------------------------------------------------------------+
enum ENUM_GARCH_REGIME
{
   REGIME_LOW,      // < 0.0008
   REGIME_NORMAL,   // 0.0008 – 0.0015
   REGIME_HIGH,     // 0.0015 – 0.0025
   REGIME_EXTREME   // > 0.0025
};

ENUM_GARCH_REGIME GetGARCHRegime(double garch_vol)
{
   if(garch_vol < 0.0008) return REGIME_LOW;
   if(garch_vol < 0.0015) return REGIME_NORMAL;
   if(garch_vol < 0.0025) return REGIME_HIGH;
   return REGIME_EXTREME;
}
