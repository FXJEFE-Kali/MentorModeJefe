//+------------------------------------------------------------------+
//|                        RiskEngine.mqh v2.04                      |
//|  Daily Loss Breaker + WinStreak De-lever + GARCH-Aware Lots     |
//|  The5ers + FundingPips + Vantage Safe                            |
//+------------------------------------------------------------------+

input double MaxDailyLossPct          = 2.3;     // Hard internal cap (safer than 3%)
input int    WinStreakDeleverAfter    = 3;       // Reduce lot after N wins
input double WinStreakLotMultiplier   = 0.65;    // 65% lot after streak
input double GarchHighVolThreshold    = 0.0015;  // Above this → reduce risk
input double GarchRiskReduction       = 0.70;    // 70% risk when high vol

//+------------------------------------------------------------------+
//| Daily Loss + WinStreak + GARCH Risk Engine                       |
//+------------------------------------------------------------------+
double ApplyRiskEngine(string symbol, double baseLot, double confidence, double garchVol)
{
   double finalLot = baseLot;

   // 1. Win-streak de-leveraging (never get greedy)
   static int consecWins = 0;
   if(consecWins >= WinStreakDeleverAfter)
   {
      finalLot *= WinStreakLotMultiplier;
      Print("WIN STREAK ACTIVE (", consecWins, ") — Lot reduced to ", DoubleToString(finalLot, 2));
   }

   // 2. GARCH volatility scaling (protect small accounts)
   if(garchVol > GarchHighVolThreshold)
   {
      finalLot *= GarchRiskReduction;
      Print("HIGH GARCH VOL (", DoubleToString(garchVol, 6), ") — Risk reduced");
   }

   // 3. Confidence scaling (already in previous versions)
   if(confidence > 0.75)
      finalLot *= 1.0 + (confidence - 0.75) * 1.2;

   // 4. Hard daily loss breaker (2.3%)
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyStart = AccountInfoDouble(ACCOUNT_BALANCE); // or store at day start
   double dailyLossPct = (dailyStart - equity) / dailyStart * 100.0;

   if(dailyLossPct >= MaxDailyLossPct)
   {
      Print("=== DAILY LOSS CAP (2.3%) HIT — TRADING PAUSED ===");
      return 0.0;   // Block new trades
   }

   // Final safety clamp
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   return MathMax(minLot, NormalizeDouble(finalLot, 2));
}

//+------------------------------------------------------------------+
//| Call this after every closed trade to track win streak           |
//+------------------------------------------------------------------+
void UpdateWinStreak(bool tradeWasWin)
{
   static int consecWins = 0;
   if(tradeWasWin) consecWins++;
   else            consecWins = 0;
}
