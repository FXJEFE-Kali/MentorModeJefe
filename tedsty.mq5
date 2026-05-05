#property copyright "NikoFXjefe"
#property link      "https://nikofxjefe.no"
#property version   "5.00"
#property strict
#property description "FXJEFE EA with Trippel Self Trainable-AI_Api Signal Integration"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Files\FilePipe.mqh>
#include <Math\Stat\Math.mqh>

enum ENUM_AccountSize { Acct_1K = 1000, Acct_5K = 5000, Acct_10K = 10000, Acct_100K = 100000 };
enum ENUM_PHASE_TYPE { Phase_1, Phase_2, Phase_3 };
enum ENUM_SIGNAL_MODE { AI_Only = 0, Strategies_Only = 1, Both = 2 };

struct PhaseRules {
   double profitTarget_Pct;
   double dailyDD_Pct;
   double totalDD_Pct;
   double riskPct;
   int minagulationDays;
};

struct FXJEFE_CandidateTrade {
   string pair;
   string strategyName;
   ENUM_ORDER_TYPE orderType;
   double lotSize;
   double openPrice;
   double stopLoss;
   double takeProfit;
   double momentumScore; 
};

input ENUM_AccountSize AccountSize = Acct_5K;
input ENUM_PHASE_TYPE PhaseType = Phase_1;
input double RiskPercent = 0.5;
input double PostTargetRiskPct = 0.5;
input double TotalProfitTargetPct = 8.0;
input int MinTradingDays = 5;
input bool InputAllowTrading = true;
input bool UseMicroBreakout = true;
input bool UsePullbackTrend = true;
input bool UseICTKillZone = true;
input bool UsePO3 = true;
input bool UsePsychLevels = true;
input bool UseStatArbitrage = true;
input bool UseCarryTrade = true;
input bool UseAISignals = true;
input ENUM_SIGNAL_MODE SignalMode = Both;
input string AI_API_URL = "http://127.0.0.1:8080/predict";
input string API_Key = "";
input string SecurityKey = "NikoFXjefeGrok2025";
input int MaxOpenTrades = 5;
input int MaxDailyTrades = 7;
input bool UseMaxDailyTrades = true;
input double MaxPairExposure_Pct = 30.0;
input double MaxLeverage = 50.0;
input double MaxVaR_Pct = 10.0;
input double MaxES_Pct = 7.0;
input double CircuitBreakerDrop_Pct = 2.0;
input int ATR_Period = 14;
input int EMA_Fast_Period = 12;
input int EMA_Slow_Period = 26;
input int RSI_Period = 14;
input int BB_Period = 20;
input double BB_Deviation = 2.0;
input int VWAP_Period = 48;
input double VWAP_Bands_Multiplier = 2.0;
input int Stochastic_K = 5;
input int Stochastic_D = 3;
input int Stochastic_Slow = 3;
input int MACD_Fast = 12;
input int MACD_Slow = 26;
input int MACD_Signal = 9;
input double MaxSlippagePips = 100.0;
input double PartialExitPips = 20.0;
input double TP2Pips = 40.0;
input double TPBEPips = 10.0;
input int FirstSessionStart = 0;
input int FirstSessionEnd = 24;
input int SecondSessionStart = 13;
input int SecondSessionEnd = 17;
input int NewsWindowHours = 2;
input bool NoSundayTrading = false;
input int AmsterdamTimeShift = -1;
input bool UseCSVLogging = true;
input string CSVDirectory = "FXJEFE\\Data";
input bool UseAPILogging = true;
input string APIPipeName = "FXJEFE_API_Pipe";
input double MaxCorrelation = 0.8;
input bool LabelSignals = true;
input double PipThreshold = 10.0;
input int ROC_Period = 10;
input int CCI_Period = 14;
input int Williams_Period = 14;
input int Momentum_Period = 14;
input int RV_Period = 14;
input int Chaikin_Period = 14;
input int ADX_Period = 14;
input int RVI_Period = 14;
input int OBV_Period = 14;
input int Volume_Delta_Period = 14;
input int AD_Period = 14;
input int Vol_Osc_Fast_Period = 14;
input int Vol_Osc_Slow_Period = 28;
input int Supertrend_Period = 10;
input double Supertrend_Multiplier = 3.0;
input int HMA_Period = 9;
input int Ichimoku_Tenkan_Period = 9;
input double SAR_Step = 0.02;
input double SAR_Maximum = 0.2;
input int DPO_Period = 20;

string dynamicPairList[] = {"EURUSD", "USDJPY", "XAUUSD", "AUDUSD", "GBPUSD", "USDCAD"};
int totalPairs = 6;
int atrHandles[], emaFastHandles[], emaSlowHandles[], rsiHandles[], bbHandles[], stochasticHandles[], macdHandles[], adxHandles[];
int cciHandles[], willrHandles[], momHandles[], obvHandles[], sarHandles[], rviHandles[], ichimokuHandles[], dpoHandles[];
double cachedATR[], cachedEMAFast[], cachedEMASlow[], cachedRSI[], cachedBBUpper[], cachedBBLower[];
double cachedStochK[], cachedStochD[], cachedMACD[], cachedMACDSignal[];
double cachedVWAP[], cachedVWAPUpper[], cachedVWAPLower[];
double garchVolatility[];
double garchAlpha = 0.1, garchBeta = 0.85, garchOmega = 0.00001;
double cachedCCI[], cachedWillR[], cachedMom[], cachedOBV[], cachedSAR[], cachedRVI[], cachedIchimokuTenkan[], cachedDPO[];
double cachedROC[], cachedRealizedVol[], cachedChaikinVol[], cachedVolumeDelta[], cachedADLine[], cachedVolOsc[];
double cachedSupertrend[], cachedHMA[];
double g_maxBalance, g_dailyStartEquity, g_initialBalance, g_lastEquityCheck;
datetime g_lastEquityTime, last_api_call[], g_lastTradeTime, lastHistoryCheck = 0;
int g_lastDealCount, g_consecutiveLosses, g_dailyTradesCount;
bool g_apiPipeOpen = false;
bool g_timerRunning = false;
CTrade trade;
CSymbolInfo symbolInfo;
CPositionInfo positionInfo;
CFilePipe apiPipe;
double lastPrice[];
datetime lastTime[];
string last_good_signal[];
datetime last_good_signal_time[];
bool indicatorsInitialized = false;
bool tradingEnabled = InputAllowTrading;
double g_totalProfitTarget = 0.0;
double g_dailyProfit = 0.0;
datetime g_lastDayReset = 0;
int g_tradingDaysCount = 0;
bool g_tradingDayActive = false;
string g_tradingDays[];
double g_maxDailyLoss = 100.0;
bool g_profitTargetReached = false;
double tokyoHigh[], tokyoLow[];
double cachedADX[];
datetime lastKillZoneStart[];
double firstHigh[], firstLow[];
double g_previousDayEquity;
bool g_useAISignals = UseAISignals; // Global variable to track AI signal state

bool CheckSecurityKey() {
   if (SecurityKey != "NikoFXjefeGrok2025") {
      Print("Security Key mismatch. EA disabled.");
      return false;
   }
   return true;
}

string ArrayToStringCustom(string &arr[]) {
   string result = "";
   for (int i = 0; i < ArraySize(arr); i++) {
      result += arr[i];
      if (i < ArraySize(arr) - 1) result += ", ";
   }
   return result;
}

double GetMomentumScore(string pair, int idx) {
   if (idx < 0 || idx >= totalPairs || ArraySize(cachedRSI) <= idx) return 0.5;
   double score = 0.5;
   if (cachedRSI[idx] > 75 || cachedRSI[idx] < 25) score += 0.3;
   if (MathAbs(cachedMACD[idx] - cachedMACDSignal[idx]) > cachedATR[idx] * 0.5) score += 0.2;
   return MathMin(score, 0.9);
}

bool NewBar(ENUM_TIMEFRAMES tf) {
   static datetime lastBar = 0;
   datetime currentBar = iTime(Symbol(), tf, 0);
   if (currentBar != lastBar) {
      lastBar = currentBar;
      return true;
   }
   return false;
}

int ArraySearchString(string &arr[], string value) {
   for (int i = 0; i < ArraySize(arr); i++) {
      if (arr[i] == value) return i;
   }
   return -1;
}

double CalculateLotSize(string pair, double slPrice, double openPrice, ENUM_ORDER_TYPE orderType) {
   if (!symbolInfo.Name(pair)) return SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN);
   double riskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * (RiskPercent / 100.0);
   double pipValue = SymbolInfoDouble(pair, SYMBOL_TRADE_TICK_VALUE);
   double slDistance = MathAbs(openPrice - slPrice) / SymbolInfoDouble(pair, SYMBOL_POINT);
   if (slDistance == 0) return SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN);
   double lotSize = riskAmount / (slDistance * pipValue);
   double minLot = SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(pair, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(pair, SYMBOL_VOLUME_STEP);
   double maxRiskLot = AccountInfoDouble(ACCOUNT_BALANCE) * 0.01 / (slDistance * pipValue);
   lotSize = MathMin(lotSize, maxRiskLot);
   int steps = (int)MathFloor((lotSize - minLot) / stepLot);
   lotSize = minLot + steps * stepLot;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   Print("Symbol: ", pair, " MinLot: ", minLot, " MaxLot: ", maxLot, " StepLot: ", stepLot, " CalculatedLot: ", lotSize);
   return lotSize;
}

bool IsHighVolatility(string pair, int idx) {
   if (idx < 0 || idx >= totalPairs || atrHandles[idx] == INVALID_HANDLE) {
      Print("Invalid ATR handle or index for ", pair, ". Assuming low volatility.");
      return false;
   }
   double atr = cachedATR[idx];
   double min_atr = SymbolInfoDouble(pair, SYMBOL_POINT) * 10;
   if (atr < min_atr) {
      Print("ATR too low for ", pair, ": ", atr, ". Skipping trade.");
      return true;
   }
   double atrValues[];
   ArraySetAsSeries(atrValues, true);
   if (CopyBuffer(atrHandles[idx], 0, 1, 1, atrValues) <= 0) {
      Print("Failed to copy ATR for ", pair, ". Error: ", GetLastError());
      return false;
   }
   double atrAvg = atrValues[0];
   return atr > atrAvg * 2;
}

double CalculateTrueLeverage(double additionalLots, string pair) {
   if (!symbolInfo.Name(pair)) return 0.0;
   double marginUsed = AccountInfoDouble(ACCOUNT_MARGIN);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double contractSize = symbolInfo.ContractSize();
   double tickValue = symbolInfo.TickValue();
   double additionalMargin = additionalLots * contractSize * SymbolInfoDouble(pair, SYMBOL_BID) / tickValue;
   return (marginUsed + additionalMargin) / equity;
}

double GetDynamicSlippage(string sym, int idx) {
   if (idx < 0 || idx >= totalPairs || ArraySize(cachedATR) <= idx) {
      Print("Invalid index ", idx, " for slippage in ", sym);
      return MaxSlippagePips;
   }
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   double atrPips = cachedATR[idx] / point;
   double slippage = MathMin(MaxSlippagePips * 10, atrPips * 2.5);
   Print("Slippage for ", sym, ": ", slippage, " points (ATR: ", atrPips, ")");
   return slippage;
}

bool CheckLiquidity(string pair) {
   if (!symbolInfo.Name(pair)) return false;
   double bid = SymbolInfoDouble(pair, SYMBOL_BID);
   double ask = SymbolInfoDouble(pair, SYMBOL_ASK);
   if (bid <= 0 || ask <= 0) {
      Print("Invalid bid/ask for ", pair, ". Assuming insufficient liquidity.");
      return false;
   }
   return true;
}

double GetAdjustedCarryRate(string pair) {
   if (!symbolInfo.Name(pair)) return 0.0;
   double swapLong = SymbolInfoDouble(pair, SYMBOL_SWAP_LONG);
   double swapShort = SymbolInfoDouble(pair, SYMBOL_SWAP_SHORT);
   MqlDateTime timeStruct;
   TimeToStruct(TimeCurrent(), timeStruct);
   if (timeStruct.day_of_week == 3) {
      swapLong *= 3.0;
      swapShort *= 3.0;
   }
   return (swapLong > swapShort) ? swapLong : swapShort;
}

// Indicator Calculation Functions
double CalculateROC(string sym, int period) {
   double prices[];
   ArraySetAsSeries(prices, true);
   if (CopyClose(sym, PERIOD_M15, 0, period + 1, prices) < period + 1) {
      Print("Failed to copy prices for ROC: ", sym);
      return 0.0;
   }
   return ((prices[0] - prices[period]) / prices[period]) * 100;
}

double CalculateStochastic(string sym, int k_period) {
   double k[];
   ArraySetAsSeries(k, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && stochasticHandles[idx] != INVALID_HANDLE && CopyBuffer(stochasticHandles[idx], 0, 0, 1, k) > 0) return k[0];
   return 50.0;
}

double CalculateCCI(string sym, int period) {
   double cci[];
   ArraySetAsSeries(cci, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && cciHandles[idx] != INVALID_HANDLE && CopyBuffer(cciHandles[idx], 0, 0, 1, cci) > 0) return cci[0];
   return 0.0;
}

double CalculateWilliams(string sym, int period) {
   double willr[];
   ArraySetAsSeries(willr, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && willrHandles[idx] != INVALID_HANDLE && CopyBuffer(willrHandles[idx], 0, 0, 1, willr) > 0) return willr[0];
   return -50.0;
}

double CalculateMomentum(string sym, int period) {
   double mom[];
   ArraySetAsSeries(mom, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && momHandles[idx] != INVALID_HANDLE && CopyBuffer(momHandles[idx], 0, 0, 1, mom) > 0) return mom[0];
   return 0.0;
}

double CalculateRealizedVol(string sym, int period) {
   double prices[];
   ArraySetAsSeries(prices, true);
   if (CopyClose(sym, PERIOD_M15, 0, period + 1, prices) < period + 1) return 0.0;
   double returns[];
   ArrayResize(returns, period);
   for (int i = 0; i < period; i++) returns[i] = MathLog(prices[i] / prices[i + 1]);
   double mean = 0.0;
   for (int i = 0; i < period; i++) mean += returns[i];
   mean /= period;
   double variance = 0.0;
   for (int i = 0; i < period; i++) variance += MathPow(returns[i] - mean, 2);
   variance /= period;
   return MathSqrt(variance) * MathSqrt(252) * 100;
}

double CalculateChaikinVol(string sym, int period) {
   double high[], low[];
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   if (CopyHigh(sym, PERIOD_M15, 0, period, high) < period || CopyLow(sym, PERIOD_M15, 0, period, low) < period) return 0.0;
   double range[];
   ArrayResize(range, period);
   for (int i = 0; i < period; i++) range[i] = high[i] - low[i];
   double ema_fast = 0.0, ema_slow = 0.0;
   double alpha_fast = 2.0 / (10 + 1), alpha_slow = 2.0 / (period + 1);
   for (int i = period - 1; i >= 0; i--) {
      ema_fast = alpha_fast * range[i] + (1 - alpha_fast) * ema_fast;
      ema_slow = alpha_slow * range[i] + (1 - alpha_slow) * ema_slow;
   }
   return (ema_slow != 0) ? (ema_fast - ema_slow) / ema_slow * 100 : 0;
}

double CalculateADX(string sym, int period) {
   double adx[];
   ArraySetAsSeries(adx, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && adxHandles[idx] != INVALID_HANDLE && CopyBuffer(adxHandles[idx], 0, 0, 1, adx) > 0) return adx[0];
   return 25.0;
}

double CalculateRVI(string sym, int period) {
   double rvi[];
   ArraySetAsSeries(rvi, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && rviHandles[idx] != INVALID_HANDLE && CopyBuffer(rviHandles[idx], 0, 0, 1, rvi) > 0) return rvi[0];
   return 0.0;
}

double CalculateOBV(string sym, int period) {
   double obv[];
   ArraySetAsSeries(obv, true);
   int idx = ArraySearchString(dynamicPairList, sym);
   if (idx >= 0 && obvHandles[idx] != INVALID_HANDLE && CopyBuffer(obvHandles[idx], 0, 0, 1, obv) > 0) return obv[0];
   return 0.0;
}

double CalculateVolumeDelta(string sym, int period) {
   long volume[];
   ArraySetAsSeries(volume, true);
   if (CopyTickVolume(sym, PERIOD_M15, 0, period, volume) < period) return 0.0;
   double delta = 0.0;
   for (int i = 0; i < period; i++) {
      double close[];
      ArraySetAsSeries(close, true);
      if (CopyClose(sym, PERIOD_M15, i, 2, close) < 2) continue;
      delta += (close[0] > close[1]) ? (double)volume[i] : -(double)volume[i]; // Explicit cast to double
   }
   return delta;
}

double CalculateADLine(string sym, int period) {
   double close[], high[], low[];
   long volume[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(volume, true);
   if (CopyClose(sym, PERIOD_M15, 0, period, close) < period ||
       CopyHigh(sym, PERIOD_M15, 0, period, high) < period ||
       CopyLow(sym, PERIOD_M15, 0, period, low) < period ||
       CopyTickVolume(sym, PERIOD_M15, 0, period, volume) < period) return 0.0;
   double ad = 0.0;
   for (int i = 0; i < period; i++) {
      double mfm = (high[i] != low[i]) ? ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) : 0;
      ad += mfm * volume[i];
   }
   return ad;
}

double CalculateVolOsc(string sym, int fast_period, int slow_period) {
   long volume[];
   ArraySetAsSeries(volume, true);
   if (CopyTickVolume(sym, PERIOD_M15, 0, slow_period, volume) < slow_period) return 0.0;
   double ema_fast = 0.0, ema_slow = 0.0;
   double alpha_fast = 2.0 / (fast_period + 1), alpha_slow = 2.0 / (slow_period + 1);
   for (int i = slow_period - 1; i >= 0; i--) {
      ema_fast = alpha_fast * volume[i] + (1 - alpha_fast) * ema_fast;
      ema_slow = alpha_slow * volume[i] + (1 - alpha_slow) * ema_slow;
   }
   return (ema_slow != 0) ? (ema_fast - ema_slow) / ema_slow * 100 : 0;
}

double CalculateSupertrend(string sym, int period, double multiplier) {
   double high[], low[], close[];
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   if (CopyHigh(sym, PERIOD_M15, 0, period + 1, high) < period + 1 ||
       CopyLow(sym, PERIOD_M15, 0, period + 1, low) < period + 1 ||
       CopyClose(sym, PERIOD_M15, 0, period + 1, close) < period + 1) return 0.0;
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(sym, PERIOD_M15, period);
   if (CopyBuffer(atr_handle, 0, 0, period + 1, atr) <= 0) return 0.0;
   double upper_band = (high[0] + low[0]) / 2 + multiplier * atr[0];
   double lower_band = (high[0] + low[0]) / 2 - multiplier * atr[0];
   return (close[0] > upper_band) ? lower_band : upper_band;
}

double CalculateHMA(string sym, int period) {
   double close[];
   ArraySetAsSeries(close, true);
   if (CopyClose(sym, PERIOD_M15, 0, period * 2, close) < period * 2) return 0.0;
   double wma1[], wma2[];
   ArrayResize(wma1, period);
   ArrayResize(wma2, period);
   for (int i = 0; i < period; i++) {
      double sum = 0.0, weight = 0.0;
      for (int j = 0; j < period / 2; j++) {
         sum += close[i + j] * (period / 2 - j);
         weight += (period / 2 - j);
      }
      wma1[i] = (weight > 0) ? sum / weight : 0.0;
      sum = 0.0; weight = 0.0;
      for (int j = 0; j < period; j++) {
         sum += close[i + j] * (period - j);
         weight += (period - j);
      }
      wma2[i] = (weight > 0) ? sum / weight : 0.0;
   }
   double diff[];
   ArrayResize(diff, period);
   for (int i = 0; i < period; i++) diff[i] = 2 * wma1[i] - wma2[i];
   double hma = 0.0;
   int sqrt_period = (int)MathSqrt(period);
   double sum = 0.0, weight = 0.0;
   for (int i = 0; i < sqrt_period; i++) {
      sum += diff[i] * (sqrt_period - i);
      weight += (sqrt_period - i);
   }
   return (weight > 0) ? sum / weight : 0.0;
}

void CalculateVWAP(int idx, string symbol) {
   if (idx < 0 || idx >= totalPairs || ArraySize(cachedVWAP) <= idx) return;
   double high[], low[], close[];
   long volumes[];
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(volumes, true);
   if (CopyHigh(symbol, PERIOD_M15, 0, VWAP_Period, high) < VWAP_Period ||
       CopyLow(symbol, PERIOD_M15, 0, VWAP_Period, low) < VWAP_Period ||
       CopyClose(symbol, PERIOD_M15, 0, VWAP_Period, close) < VWAP_Period ||
       CopyTickVolume(symbol, PERIOD_M15, 0, VWAP_Period, volumes) < VWAP_Period) {
      Print("Failed to copy VWAP data for ", symbol);
      return;
   }
   double sumPriceVolume = 0.0, sumVolume = 0.0;
   double typicalPrices[];
   ArrayResize(typicalPrices, VWAP_Period);
   for (int i = 0; i < VWAP_Period; i++) {
      double typicalPrice = (high[i] + low[i] + close[i]) / 3.0;
      sumPriceVolume += typicalPrice * (double)volumes[i];
      sumVolume += (double)volumes[i];
      typicalPrices[i] = typicalPrice;
   }
   if (sumVolume > 0) {
      cachedVWAP[idx] = sumPriceVolume / sumVolume;
      double sumVariance = 0.0;
      for (int i = 0; i < VWAP_Period; i++) {
         sumVariance += MathPow(typicalPrices[i] - cachedVWAP[idx], 2) * (double)volumes[i];
      }
      double variance = sumVariance / sumVolume;
      double std_dev = MathSqrt(variance);
      cachedVWAPUpper[idx] = cachedVWAP[idx] + VWAP_Bands_Multiplier * std_dev;
      cachedVWAPLower[idx] = cachedVWAP[idx] - VWAP_Bands_Multiplier * std_dev;
   }
}

void ValidateDynamicPairList() {
   for (int i = ArraySize(dynamicPairList) - 1; i >= 0; i--) {
      if (!SymbolInfoDouble(dynamicPairList[i], SYMBOL_BID)) {
         Print("Removing invalid symbol: ", dynamicPairList[i]);
         ArrayRemove(dynamicPairList, i);
      }
   }
   totalPairs = ArraySize(dynamicPairList);
   if (totalPairs == 0) {
      Print("No valid symbols in dynamicPairList. EA will not trade.");
      tradingEnabled = false;
   }
}

void ArrayRemove(string &arr[], int index) {
   int size = ArraySize(arr);
   if (index < 0 || index >= size) return;
   string temp[];
   ArrayResize(temp, size - 1);
   for (int i = 0; i < index; i++) temp[i] = arr[i];
   for (int i = index + 1; i < size; i++) temp[i - 1] = arr[i];
   ArrayCopy(arr, temp);
}

bool IsTradingAllowedNow() {
   MqlDateTime timeStruct;
   TimeToStruct(TimeGMT(), timeStruct);
   int hour = timeStruct.hour;
   if (NoSundayTrading && timeStruct.day_of_week == 0) return false;
   if (hour >= 15 && hour < 17) return false;
   return (hour >= FirstSessionStart && hour < FirstSessionEnd) ||
          (hour >= SecondSessionStart && hour < SecondSessionEnd);
}

void UpdateGARCHVolatility() {
   for (int i = 0; i < totalPairs; i++) {
      if (!symbolInfo.Name(dynamicPairList[i])) continue;
      double price = SymbolInfoDouble(dynamicPairList[i], SYMBOL_BID);
      if (lastPrice[i] > 0 && price > 0) {
         double ret = MathLog(price / lastPrice[i]);
         garchVolatility[i] = MathSqrt(garchOmega + garchAlpha * MathPow(ret, 2) + garchBeta * MathPow(garchVolatility[i], 2));
      }
      lastPrice[i] = price;
   }
}

void MultiPartialExit() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (PositionSelectByTicket(ticket)) {
         string sym = PositionGetString(POSITION_SYMBOL);
         if (!symbolInfo.Name(sym)) continue;
         int idx = ArraySearchString(dynamicPairList, sym);
         if (idx < 0) continue;
         int type = (int)PositionGetInteger(POSITION_TYPE);
         double op = PositionGetDouble(POSITION_PRICE_OPEN);
         double vol = PositionGetDouble(POSITION_VOLUME);
         double sl = PositionGetDouble(POSITION_SL);
         double tp = PositionGetDouble(POSITION_TP);
         double currentPrice = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(sym, SYMBOL_BID) : SymbolInfoDouble(sym, SYMBOL_ASK);
         double pts = SymbolInfoDouble(sym, SYMBOL_POINT);
         double pips = (type == POSITION_TYPE_BUY) ? (currentPrice - op) / pts : (op - currentPrice) / pts;

         // Partial exit at first target
         if (pips >= PartialExitPips && pips < TP2Pips && vol > 0) {
            double closeVolume = NormalizeDouble(vol / 3.0, 2);
            if (closeVolume >= SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN)) {
               if (trade.PositionClosePartial(ticket, closeVolume)) {
                  double newSL = (type == POSITION_TYPE_BUY) ? op + (TPBEPips * pts) : op - (TPBEPips * pts);
                  trade.PositionModify(ticket, newSL, tp);
                  Print("TP1 hit: Closed 1/3 of #", ticket, ", SL to BE + ", TPBEPips, " pips");
               }
            }
         }
         // Second partial exit
         else if (pips >= TP2Pips && vol > 0) {
            double closeVolume = NormalizeDouble(vol / 2.0, 2);
            if (closeVolume >= SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN)) {
               if (trade.PositionClosePartial(ticket, closeVolume)) {
                  Print("TP2 hit: Closed half of #", ticket);
               }
            }
         }
         // Trailing stop
         if (pips >= TPBEPips && vol > 0) {
            double trailingStop = (type == POSITION_TYPE_BUY) ? currentPrice - cachedATR[idx] * 1.5 : currentPrice + cachedATR[idx] * 1.5;
            if ((type == POSITION_TYPE_BUY && trailingStop > sl) || (type == POSITION_TYPE_SELL && trailingStop < sl)) {
               if (trade.PositionModify(ticket, trailingStop, tp)) {
                  Print("Trailing stop updated for #", ticket, " to ", trailingStop);
               }
            }
         }
      }
   }
}

bool HasMicroBreakoutSignal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UseMicroBreakout) return false;
   if (!symbolInfo.Name(pair)) return false;
   double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
   double filter = 0.3 * cachedATR[idx];
   MqlDateTime timeStruct;
   TimeToStruct(TimeCurrent(), timeStruct);
   int hour = timeStruct.hour;
   if (hour >= 8 && hour < 17) {
      if (tokyoHigh[idx] == 0 || tokyoLow[idx] == 0) {
         // Initialize Tokyo session high/low (assumed from 00:00-08:00 Tokyo time)
         double high[], low[];
         ArraySetAsSeries(high, true);
         ArraySetAsSeries(low, true);
         if (CopyHigh(pair, PERIOD_M15, 0, 32, high) > 0 && CopyLow(pair, PERIOD_M15, 0, 32, low) > 0) {
            tokyoHigh[idx] = high[0];
            tokyoLow[idx] = low[0];
            for (int j = 1; j < 32; j++) {
               tokyoHigh[idx] = MathMax(tokyoHigh[idx], high[j]);
               tokyoLow[idx] = MathMin(tokyoLow[idx], low[j]);
            }
         }
         return false;
      }
      if (currentPrice > tokyoHigh[idx] + filter) {
         c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
         c.stopLoss = tokyoLow[idx]; c.takeProfit = currentPrice + (currentPrice - tokyoLow[idx]) * 2;
         c.strategyName = "MicroBreakout"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      } else if (currentPrice < tokyoLow[idx] - filter) {
         c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
         c.stopLoss = tokyoHigh[idx]; c.takeProfit = currentPrice - (tokyoHigh[idx] - currentPrice) * 2;
         c.strategyName = "MicroBreakout"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      }
   }
   return false;
}

bool HasPullbackTrendSignal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UsePullbackTrend) return false;
   if (!symbolInfo.Name(pair)) return false;
   double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
   if (cachedEMAFast[idx] > cachedEMASlow[idx] && cachedADX[idx] > 30) {
      if (currentPrice <= cachedBBLower[idx]) {
         c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
         c.stopLoss = currentPrice - cachedATR[idx] * 2; c.takeProfit = currentPrice + cachedATR[idx] * 3;
         c.strategyName = "PullbackTrend"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      }
   } else if (cachedEMAFast[idx] < cachedEMASlow[idx] && cachedADX[idx] > 30) {
      if (currentPrice >= cachedBBUpper[idx]) {
         c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
         c.stopLoss = currentPrice + cachedATR[idx] * 2; c.takeProfit = currentPrice - cachedATR[idx] * 3;
         c.strategyName = "PullbackTrend"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      }
   }
   return false;
}

bool HasICTKillZoneSignal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UseICTKillZone) return false;
   if (!symbolInfo.Name(pair)) return false;
   MqlDateTime timeStruct;
   TimeToStruct(TimeCurrent(), timeStruct);
   int hour = timeStruct.hour;
   if ((hour >= 6 && hour < 10) || (hour >= 11 && hour < 15)) {
      double prevHigh = iHigh(pair, PERIOD_M15, 1);
      double prevLow = iLow(pair, PERIOD_M15, 1);
      double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
      if (currentPrice > prevHigh && prevHigh > 0) {
         c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
         c.stopLoss = prevLow - cachedATR[idx]; c.takeProfit = currentPrice + 2 * cachedATR[idx];
         c.strategyName = "ICTKillZone"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      } else if (currentPrice < prevLow && prevLow > 0) {
         c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
         c.stopLoss = prevHigh + cachedATR[idx]; c.takeProfit = currentPrice - 2 * cachedATR[idx];
         c.strategyName = "ICTKillZone"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      }
   }
   return false;
}

bool HasPO3Signal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UsePO3) return false;
   if (!symbolInfo.Name(pair)) return false;
   MqlDateTime timeStruct;
   TimeToStruct(TimeCurrent(), timeStruct);
   int hour = timeStruct.hour;
   if (hour >= 7 && hour < 10) {
      if (firstHigh[idx] == 0 || firstLow[idx] == 0) {
         // Initialize first high/low (assumed from 00:00-07:00)
         double high[], low[];
         ArraySetAsSeries(high, true);
         ArraySetAsSeries(low, true);
         if (CopyHigh(pair, PERIOD_M15, 0, 28, high) > 0 && CopyLow(pair, PERIOD_M15, 0, 28, low) > 0) {
            firstHigh[idx] = high[0];
            firstLow[idx] = low[0];
            for (int j = 1; j < 28; j++) {
               firstHigh[idx] = MathMax(firstHigh[idx], high[j]);
               firstLow[idx] = MathMin(firstLow[idx], low[j]);
            }
         }
         return false;
      }
      double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
      if (currentPrice > firstHigh[idx] && firstHigh[idx] > 0) {
         c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
         c.stopLoss = firstLow[idx]; c.takeProfit = currentPrice + 1.5 * cachedATR[idx];
         c.strategyName = "PO3"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      } else if (currentPrice < firstLow[idx] && firstLow[idx] > 0) {
         c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
         c.stopLoss = firstHigh[idx]; c.takeProfit = currentPrice - 1.5 * cachedATR[idx];
         c.strategyName = "PO3"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      }
   }
   return false;
}

bool HasPsychLevelsSignal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UsePsychLevels) return false;
   if (!symbolInfo.Name(pair)) return false;
   double psychLevel = MathRound(SymbolInfoDouble(pair, SYMBOL_BID) / 0.01) * 0.01;
   double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
   if (MathAbs(currentPrice - psychLevel) <= 0.5 * cachedATR[idx]) {
      if (cachedRSI[idx] < 25) {
         c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
         c.stopLoss = currentPrice - 2 * cachedATR[idx]; c.takeProfit = currentPrice + 3 * cachedATR[idx];
         c.strategyName = "PsychLevels"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      } else if (cachedRSI[idx] > 75) {
         c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
         c.stopLoss = currentPrice + 2 * cachedATR[idx]; c.takeProfit = currentPrice - 3 * cachedATR[idx];
         c.strategyName = "PsychLevels"; c.momentumScore = GetMomentumScore(pair, idx);
         c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
         return true;
      }
   }
   return false;
}

bool HasStatArbitrageSignal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UseStatArbitrage) return false;
   if (!symbolInfo.Name(pair)) return false;
   double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
   if (currentPrice < cachedBBLower[idx] && currentPrice < cachedVWAP[idx]) {
      c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
      c.stopLoss = currentPrice - 2 * cachedATR[idx]; c.takeProfit = cachedBBUpper[idx];
      c.strategyName = "StatArbitrage"; c.momentumScore = GetMomentumScore(pair, idx);
      c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
      return true;
   } else if (currentPrice > cachedBBUpper[idx] && currentPrice > cachedVWAP[idx]) {
      c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
      c.stopLoss = currentPrice + 2 * cachedATR[idx]; c.takeProfit = cachedBBLower[idx];
      c.strategyName = "StatArbitrage"; c.momentumScore = GetMomentumScore(pair, idx);
      c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
      return true;
   }
   return false;
}

bool HasCarryTradeSignal(string pair, int idx, FXJEFE_CandidateTrade &c) {
   if (!UseCarryTrade) return false;
   if (!symbolInfo.Name(pair)) return false;
   double swapLong = SymbolInfoDouble(pair, SYMBOL_SWAP_LONG);
   double swapShort = SymbolInfoDouble(pair, SYMBOL_SWAP_SHORT);
   double currentPrice = SymbolInfoDouble(pair, SYMBOL_BID);
   double carryRate = GetAdjustedCarryRate(pair);
   if (swapLong > 0 && currentPrice > cachedEMASlow[idx] && carryRate > 0.1) {
      c.pair = pair; c.orderType = ORDER_TYPE_BUY; c.openPrice = currentPrice;
      c.stopLoss = cachedEMASlow[idx] - cachedATR[idx]; c.takeProfit = currentPrice + 2 * cachedATR[idx];
      c.strategyName = "CarryTrade"; c.momentumScore = GetMomentumScore(pair, idx);
      c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
      return true;
   } else if (swapShort > 0 && currentPrice < cachedEMASlow[idx] && carryRate > 0.1) {
      c.pair = pair; c.orderType = ORDER_TYPE_SELL; c.openPrice = currentPrice;
      c.stopLoss = cachedEMASlow[idx] + cachedATR[idx]; c.takeProfit = currentPrice - 2 * cachedATR[idx];
      c.strategyName = "CarryTrade"; c.momentumScore = GetMomentumScore(pair, idx);
      c.lotSize = CalculateLotSize(pair, c.stopLoss, c.openPrice, c.orderType);
      return true;
   }
   return false;
}

string CallAIAPI(string symbol, int idx) {
   if (!g_useAISignals || !indicatorsInitialized) {
      Print("AI signals disabled or indicators not initialized for ", symbol);
      return last_good_signal[idx];
   }
   if (TimeCurrent() - last_api_call[idx] < 5) {
      Print("API call skipped for ", symbol, " (cooldown)");
      return last_good_signal[idx];
   }
   if (!symbolInfo.Name(symbol)) {
      Print("Invalid symbol ", symbol, " for API call");
      return last_good_signal[idx];
   }

   double price = SymbolInfoDouble(symbol, SYMBOL_BID);
   double atr = cachedATR[idx] > 0 ? cachedATR[idx] : 0.001;
   double ema_diff = (MathIsValidNumber(cachedEMAFast[idx]) && MathIsValidNumber(cachedEMASlow[idx])) ? cachedEMAFast[idx] - cachedEMASlow[idx] : 0.0;
   double rsi = (cachedRSI[idx] >= 0 && cachedRSI[idx] <= 100) ? cachedRSI[idx] : 50.0;
   double garch_vol = garchVolatility[idx] >= 0 ? garchVolatility[idx] : 0.0;
   double macd_diff = (MathIsValidNumber(cachedMACD[idx]) && MathIsValidNumber(cachedMACDSignal[idx])) ? cachedMACD[idx] - cachedMACDSignal[idx] : 0.0;
   double vwap = cachedVWAP[idx] > 0 ? cachedVWAP[idx] : price;
   double price_vwap_diff = price - vwap;
   double bb_position = (cachedBBUpper[idx] > cachedBBLower[idx] && cachedBBUpper[idx] > 0 && cachedBBLower[idx] > 0) ? 
                       (price - cachedBBLower[idx]) / (cachedBBUpper[idx] - cachedBBLower[idx]) : 0.5;
   double roc = CalculateROC(symbol, ROC_Period);
   double stochastic = CalculateStochastic(symbol, Stochastic_K);
   double cci = CalculateCCI(symbol, CCI_Period);
   double williams = CalculateWilliams(symbol, Williams_Period);
   double momentum = CalculateMomentum(symbol, Momentum_Period);
   double realized_vol = CalculateRealizedVol(symbol, RV_Period);
   double chaikin_vol = CalculateChaikinVol(symbol, Chaikin_Period);
   double adx = CalculateADX(symbol, ADX_Period);
   double rvi = CalculateRVI(symbol, RVI_Period);
   double obv = CalculateOBV(symbol, OBV_Period);
   double volume_delta = CalculateVolumeDelta(symbol, Volume_Delta_Period);
   double ad_line = CalculateADLine(symbol, AD_Period);
   double vol_osc = CalculateVolOsc(symbol, Vol_Osc_Fast_Period, Vol_Osc_Slow_Period);
   double supertrend = CalculateSupertrend(symbol, Supertrend_Period, Supertrend_Multiplier);
   double hma = CalculateHMA(symbol, HMA_Period);
   double ichimoku_tenkan = cachedIchimokuTenkan[idx] > 0 ? cachedIchimokuTenkan[idx] : price;
   double sar = cachedSAR[idx] > 0 ? cachedSAR[idx] : price;
   double dpo = cachedDPO[idx];
   double spread = (SymbolInfoDouble(symbol, SYMBOL_ASK) - price);
   double sentiment = 0.0; // Rely on fix_csv_encoding.py for sentiment

   string json = StringFormat(
      "{\"symbol\":\"%s\",\"price\":%.5f,\"atr\":%.8f,\"ema_diff\":%.8f,\"rsi\":%.2f,\"garch_vol\":%.8f,\"macd_diff\":%.8f,"
      "\"vwap\":%.5f,\"price_vwap_diff\":%.5f,\"bb_position\":%.5f,\"roc\":%.2f,\"stochastic\":%.2f,\"cci\":%.2f,"
      "\"williams\":%.2f,\"momentum\":%.2f,\"realized_vol\":%.2f,\"chaikin_vol\":%.2f,\"adx\":%.2f,\"rvi\":%.2f,"
      "\"obv\":%.0f,\"volume_delta\":%.0f,\"ad_line\":%.0f,\"vol_osc\":%.2f,\"supertrend\":%.5f,\"hma\":%.5f,"
      "\"ichimoku_tenkan\":%.5f,\"sar\":%.5f,\"dpo\":%.2f,\"spread\":%.5f,\"sentiment\":%.2f}",
      symbol, price, atr, ema_diff, rsi, garch_vol, macd_diff, vwap, price_vwap_diff, bb_position,
      roc, stochastic, cci, williams, momentum, realized_vol, chaikin_vol, adx, rvi, obv,
      volume_delta, ad_line, vol_osc, supertrend, hma, ichimoku_tenkan, sar, dpo, spread, sentiment);

   string headers = "Content-Type: application/json\r\n" + (API_Key != "" ? "Authorization: Bearer " + API_Key + "\r\n" : "");
   char post[], result[];
   StringToCharArray(json, post);
   string response_headers;
   int maxRetries = 5;
   for (int retry = 0; retry < maxRetries; retry++) {
      int res = WebRequest("POST", AI_API_URL, headers, 5000, post, result, response_headers);
      if (res == 200 && ArraySize(result) > 0) {
         string response = CharArrayToString(result);
         Print("API success for ", symbol, ": Response=", response);
         last_api_call[idx] = TimeCurrent();
         string signal = "hold";
         if (StringFind(response, "\"signal\":\"buy\"") >= 0 || StringFind(response, "buy") >= 0) signal = "buy";
         else if (StringFind(response, "\"signal\":\"sell\"") >= 0 || StringFind(response, "sell") >= 0) signal = "sell";
         last_good_signal[idx] = signal;
         last_good_signal_time[idx] = TimeCurrent();
         return signal;
      } else {
         Print("API call failed for ", symbol, ": HTTP=", res, ", Error=", GetLastError(), ", Retry ", retry + 1, "/", maxRetries);
         Sleep(1000 * (retry + 1));
      }
   }
   Print("API failed after ", maxRetries, " attempts for ", symbol, ". Disabling AI signals.");
   g_useAISignals = false; // Use global variable instead of input
   return last_good_signal[idx];
}

void LogFeatures() {
   if (!UseCSVLogging) return;
   string file_path = CSVDirectory + "\\FXJEFE_Features.csv";
   string log_file_path = CSVDirectory + "\\FXJEFE_log.txt";
   int csvHandle = FileOpen(file_path, FILE_READ | FILE_WRITE | FILE_CSV | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON, ',');
   if (csvHandle == INVALID_HANDLE) {
      Print("Failed to open CSV file: ", file_path, ", Error: ", GetLastError());
      return;
   }
   int logHandle = FileOpen(log_file_path, FILE_READ | FILE_WRITE | FILE_TXT | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON);
   if (logHandle == INVALID_HANDLE) {
      Print("Failed to open log file: ", log_file_path, ", Error: ", GetLastError());
   } else {
      FileSeek(logHandle, 0, SEEK_END);
   }
   FileSeek(csvHandle, 0, SEEK_END);
   if (FileTell(csvHandle) == 0) {
      FileWrite(csvHandle, "time,symbol,price,atr,ema_diff,rsi,garch_vol,macd_diff,vwap,price_vwap_diff,bb_position,"
                "roc,stochastic,cci,williams,momentum,realized_vol,chaikin_vol,adx,rvi,obv,volume_delta,ad_line,vol_osc,"
                "supertrend,hma,ichimoku_tenkan,sar,dpo,spread,sentiment,signal");
   }
   int rowsWritten = 0;
   for (int i = 0; i < totalPairs; i++) {
      string sym = dynamicPairList[i];
      if (!symbolInfo.Name(sym)) continue;
      double price = SymbolInfoDouble(sym, SYMBOL_BID);
      if (price <= 0) continue;
      double ema_diff = cachedEMAFast[i] - cachedEMASlow[i];
      double macd_diff = cachedMACD[i] - cachedMACDSignal[i];
      double price_vwap_diff = price - cachedVWAP[i];
      double bb_position = (cachedBBUpper[i] > cachedBBLower[i]) ? (price - cachedBBLower[i]) / (cachedBBUpper[i] - cachedBBLower[i]) : 0.5;
      double roc = CalculateROC(sym, ROC_Period);
      double stochastic = CalculateStochastic(sym, Stochastic_K);
      double cci = CalculateCCI(sym, CCI_Period);
      double williams = CalculateWilliams(sym, Williams_Period);
      double momentum = CalculateMomentum(sym, Momentum_Period);
      double realized_vol = CalculateRealizedVol(sym, RV_Period);
      double chaikin_vol = CalculateChaikinVol(sym, Chaikin_Period);
      double adx = CalculateADX(sym, ADX_Period);
      double rvi = CalculateRVI(sym, RVI_Period);
      double obv = CalculateOBV(sym, OBV_Period);
      double volume_delta = CalculateVolumeDelta(sym, Volume_Delta_Period);
      double ad_line = CalculateADLine(sym, AD_Period);
      double vol_osc = CalculateVolOsc(sym, Vol_Osc_Fast_Period, Vol_Osc_Slow_Period);
      double supertrend = CalculateSupertrend(sym, Supertrend_Period, Supertrend_Multiplier);
      double hma = CalculateHMA(sym, HMA_Period);
      double spread = SymbolInfoDouble(sym, SYMBOL_ASK) - price;
      double sentiment = 0.0; // Pipeline computes sentiment
      string timestamp = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS);
      string signal = last_good_signal[i];
      if (!MathIsValidNumber(price) || !MathIsValidNumber(cachedATR[i]) || !MathIsValidNumber(ema_diff) ||
          !MathIsValidNumber(cachedRSI[i]) || !MathIsValidNumber(garchVolatility[i]) || !MathIsValidNumber(macd_diff) ||
          !MathIsValidNumber(cachedVWAP[i]) || !MathIsValidNumber(price_vwap_diff) || !MathIsValidNumber(bb_position)) {
         Print("Invalid indicator values for ", sym);
         continue;
      }
      FileWrite(csvHandle, timestamp, sym, DoubleToString(price, 5), DoubleToString(cachedATR[i], 8),
                DoubleToString(ema_diff, 8), DoubleToString(cachedRSI[i], 2), DoubleToString(garchVolatility[i], 8),
                DoubleToString(macd_diff, 8), DoubleToString(cachedVWAP[i], 5), DoubleToString(price_vwap_diff, 5),
                DoubleToString(bb_position, 5), DoubleToString(roc, 2), DoubleToString(stochastic, 2),
                DoubleToString(cci, 2), DoubleToString(williams, 2), DoubleToString(momentum, 2),
                DoubleToString(realized_vol, 2), DoubleToString(chaikin_vol, 2), DoubleToString(adx, 2),
                DoubleToString(rvi, 2), DoubleToString(obv, 0), DoubleToString(volume_delta, 0),
                DoubleToString(ad_line, 0), DoubleToString(vol_osc, 2), DoubleToString(supertrend, 5),
                DoubleToString(hma, 5), DoubleToString(cachedIchimokuTenkan[i], 5), DoubleToString(cachedSAR[i], 5),
                DoubleToString(cachedDPO[i], 2), DoubleToString(spread, 5), DoubleToString(sentiment, 2), signal);
      rowsWritten++;
      if (logHandle != INVALID_HANDLE) {
         string logEntry = timestamp + " Features for " + sym + ": price=" + DoubleToString(price, 5) +
                           ", atr=" + DoubleToString(cachedATR[i], 8) + ", signal=" + signal;
         FileWrite(logHandle, logEntry);
      }
   }
   FileFlush(csvHandle);
   FileClose(csvHandle);
   if (logHandle != INVALID_HANDLE) {
      FileFlush(logHandle);
      FileClose(logHandle);
   }
   Print("Logged ", rowsWritten, " rows to ", file_path);
}

void LogTradeOpen(const FXJEFE_CandidateTrade &tradeData) {
   if (!UseCSVLogging) return;
   string file_path = CSVDirectory + "\\FXJEFE_trades.csv";
   int handle = FileOpen(file_path, FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON, ',');
   if (handle == INVALID_HANDLE) {
      Print("Failed to open trades CSV: ", file_path, ", Error: ", GetLastError());
      return;
   }
   FileSeek(handle, 0, SEEK_END);
   if (FileTell(handle) == 0) {
      FileWrite(handle, "positionId,timestamp,symbol,strategy,orderType,volume,price,sl,tp");
   }
   string orderTypeStr = (tradeData.orderType == ORDER_TYPE_BUY) ? "BUY" : "SELL";
   ulong ticket = trade.ResultOrder();
   if (ticket == 0) {
      Print("Failed to get ticket for trade: ", tradeData.pair, ", strategy: ", tradeData.strategyName);
   } else {
      FileWrite(handle, ticket, TimeToString(TimeCurrent()), tradeData.pair, tradeData.strategyName,
                orderTypeStr, DoubleToString(tradeData.lotSize, 2), DoubleToString(tradeData.openPrice, 5),
                DoubleToString(tradeData.stopLoss, 5), DoubleToString(tradeData.takeProfit, 5));
      Print("Logged trade: ticket=", ticket, ", symbol=", tradeData.pair, ", strategy=", tradeData.strategyName);
   }
   FileFlush(handle);
   FileClose(handle);
}

void LogTradeOutcome(ulong dealTicket, string symbol, string strategy, double profit) {
   if (!UseCSVLogging) return;
   string file_path = CSVDirectory + "\\FXJEFE_trades_outcomes.csv";
   int handle = FileOpen(file_path, FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_COMMON, ',');
   if (handle == INVALID_HANDLE) {
      Print("Failed to open outcomes CSV: ", file_path, ", Error: ", GetLastError());
      return;
   }
   FileSeek(handle, 0, SEEK_END);
   if (FileTell(handle) == 0) {
      FileWrite(handle, "dealTicket,timestamp,symbol,strategy,profit");
   }
   FileWrite(handle, dealTicket, TimeToString(TimeCurrent()), symbol, strategy, DoubleToString(profit, 2));
   FileFlush(handle);
   FileClose(handle);
   Print("Logged outcome: ticket=", dealTicket, ", symbol=", symbol, ", profit=", profit);
}

double CalculatePairCorrelation(string pair1, string pair2, int period) {
   double prices1[], prices2[];
   ArraySetAsSeries(prices1, true);
   ArraySetAsSeries(prices2, true);
   if (CopyClose(pair1, PERIOD_M15, 0, period, prices1) < period || 
       CopyClose(pair2, PERIOD_M15, 0, period, prices2) < period) {
      Print("Failed to copy prices for correlation: ", pair1, " and ", pair2);
      return 0.0;
   }
   double corr;
   if (!MathCorrelationPearson(prices1, prices2, corr)) {
      Print("Correlation calculation failed for ", pair1, " and ", pair2);
      return 0.0;
   }
   return corr;
}

bool CheckCorrelationLimit(FXJEFE_CandidateTrade &candidate) {
   for (int i = 0; i < PositionsTotal(); i++) {
      if (positionInfo.SelectByTicket(PositionGetTicket(i))) {
         string existingPair = positionInfo.Symbol();
         if (existingPair != candidate.pair) {
            double corr = CalculatePairCorrelation(candidate.pair, existingPair, 20);
            if (MathAbs(corr) > MaxCorrelation) {
               Print("Trade rejected: High correlation (", corr, ") between ", candidate.pair, " and ", existingPair);
               return false;
            }
         }
      }
   }
   return true;
}

void ScanAllStrategies(FXJEFE_CandidateTrade &candidates[]) {
   ArrayResize(candidates, 0);
   for (int i = 0; i < totalPairs; i++) {
      string sym = dynamicPairList[i];
      if (!CheckLiquidity(sym)) {
         Print("Insufficient liquidity for ", sym, ". Skipping.");
         continue;
      }
      if (IsHighVolatility(sym, i)) {
         Print("High volatility for ", sym, ". Skipping.");
         continue;
      }
      FXJEFE_CandidateTrade c;
      if (SignalMode == Strategies_Only || SignalMode == Both) {
         if (HasMicroBreakoutSignal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         if (HasPullbackTrendSignal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         if (HasICTKillZoneSignal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         if (HasPO3Signal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         if (HasPsychLevelsSignal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         if (HasStatArbitrageSignal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         if (HasCarryTradeSignal(sym, i, c) && CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
      }
      if (SignalMode == AI_Only || SignalMode == Both) {
         string aiSignal = CallAIAPI(sym, i);
         if (aiSignal != "hold") {
            c.pair = sym; c.strategyName = "AI_Signal";
            c.orderType = (aiSignal == "buy") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
            c.openPrice = (c.orderType == ORDER_TYPE_BUY) ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID);
            c.stopLoss = (c.orderType == ORDER_TYPE_BUY) ? c.openPrice - cachedATR[i] * 2 : c.openPrice + cachedATR[i] * 2;
            c.takeProfit = (c.orderType == ORDER_TYPE_BUY) ? c.openPrice + cachedATR[i] * 3 : c.openPrice - cachedATR[i] * 3;
            c.lotSize = CalculateLotSize(sym, c.stopLoss, c.openPrice, c.orderType);
            c.momentumScore = GetMomentumScore(sym, i);
            if (CheckCorrelationLimit(c)) ArrayAppend(candidates, c);
         }
      }
   }
}

void ArrayAppend(FXJEFE_CandidateTrade &arr[], FXJEFE_CandidateTrade &item) {
   int size = ArraySize(arr);
   ArrayResize(arr, size + 1);
   arr[size] = item;
}

void VerifySymbolSpecs(string pair, double lotSize) {
   if (!symbolInfo.Name(pair)) return;
   Print("Symbol: ", pair);
   Print("Min Volume: ", SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN));
   Print("Max Volume: ", SymbolInfoDouble(pair, SYMBOL_VOLUME_MAX));
   Print("Volume Step: ", SymbolInfoDouble(pair, SYMBOL_VOLUME_STEP));
   Print("Calculated Lot Size: ", lotSize);
}

void PickAndOpenBestTrades(FXJEFE_CandidateTrade &candidates[]) {
   int size = ArraySize(candidates);
   for (int i = 0; i < size - 1; i++) {
      for (int j = 0; j < size - i - 1; j++) {
         if (candidates[j].momentumScore < candidates[j + 1].momentumScore) {
            FXJEFE_CandidateTrade temp = candidates[j];
            candidates[j] = candidates[j + 1];
            candidates[j + 1] = temp;
         }
      }
   }
   int tradesToOpen = MathMin(ArraySize(candidates), MaxOpenTrades - PositionsTotal());
   for (int i = 0; i < tradesToOpen; i++) {
      if (!symbolInfo.Name(candidates[i].pair)) {
         Print("Invalid symbol ", candidates[i].pair, " for trade. Skipping.");
         continue;
      }
      VerifySymbolSpecs(candidates[i].pair, candidates[i].lotSize);
      double leverage = CalculateTrueLeverage(candidates[i].lotSize, candidates[i].pair);
      if (leverage > MaxLeverage) {
         candidates[i].lotSize *= MaxLeverage / leverage;
         Print("Adjusted lot size for ", candidates[i].pair, " to ", candidates[i].lotSize, " due to leverage limit");
      }
      double slippage = GetDynamicSlippage(candidates[i].pair, ArraySearchString(dynamicPairList, candidates[i].pair));
      trade.SetDeviationInPoints((long)(slippage / SymbolInfoDouble(candidates[i].pair, SYMBOL_POINT)));
      trade.SetTypeFilling(ORDER_FILLING_IOC);
      if (trade.PositionOpen(candidates[i].pair, candidates[i].orderType, candidates[i].lotSize,
                            candidates[i].openPrice, candidates[i].stopLoss, candidates[i].takeProfit,
                            candidates[i].strategyName)) {
         g_dailyTradesCount++;
         g_tradingDayActive = true;
         LogTradeOpen(candidates[i]);
         Print("Opened trade: ", candidates[i].pair, ", strategy=", candidates[i].strategyName,
               ", type=", (candidates[i].orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), ", lot=", candidates[i].lotSize);
      } else {
         Print("Failed to open trade: ", candidates[i].pair, ", Error: ", GetLastError());
      }
   }
}

double CalculateVaR() {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if (equity <= 0) return 0.0;
   double totalRisk = 0.0;
   for (int i = 0; i < PositionsTotal(); i++) {
      if (positionInfo.SelectByTicket(PositionGetTicket(i))) {
         string sym = positionInfo.Symbol();
         int idx = ArraySearchString(dynamicPairList, sym);
         if (idx >= 0 && symbolInfo.Name(sym)) {
            totalRisk += positionInfo.Volume() * cachedATR[idx] * SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
         }
      }
   }
   return (totalRisk / equity) * 100.0;
}

void CheckRiskLimits() {
   double var = CalculateVaR();
   if (var > MaxVaR_Pct) {
      Print("VaR limit exceeded: ", var, "% > ", MaxVaR_Pct, "%");
      tradingEnabled = false;
   }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double drawdown = (balance - equity) / balance * 100.0;
   if (drawdown > 5.0) {
      Print("Max drawdown limit exceeded: ", drawdown, "% > 5%");
      tradingEnabled = false;
   }
}

int OnInit() {
   if (!CheckSecurityKey()) return INIT_FAILED;
   ValidateDynamicPairList();
   if (totalPairs == 0) {
      Print("No valid symbols in dynamicPairList after validation. Initialization failed.");
      return INIT_FAILED;
   }

   string currentSymbol = Symbol();
   int idx = ArraySearchString(dynamicPairList, currentSymbol);
   if (idx == -1) {
      Print("Current symbol ", currentSymbol, " not in dynamicPairList. EA will monitor all pairs.");
   }

   // Resize all arrays
   ArrayResize(atrHandles, totalPairs); ArrayResize(emaFastHandles, totalPairs); ArrayResize(emaSlowHandles, totalPairs);
   ArrayResize(rsiHandles, totalPairs); ArrayResize(bbHandles, totalPairs); ArrayResize(stochasticHandles, totalPairs);
   ArrayResize(macdHandles, totalPairs); ArrayResize(adxHandles, totalPairs); ArrayResize(cciHandles, totalPairs);
   ArrayResize(willrHandles, totalPairs); ArrayResize(momHandles, totalPairs); ArrayResize(obvHandles, totalPairs);
   ArrayResize(sarHandles, totalPairs); ArrayResize(rviHandles, totalPairs); ArrayResize(ichimokuHandles, totalPairs);
   ArrayResize(dpoHandles, totalPairs); ArrayResize(cachedATR, totalPairs); ArrayResize(cachedEMAFast, totalPairs);
   ArrayResize(cachedEMASlow, totalPairs); ArrayResize(cachedRSI, totalPairs); ArrayResize(cachedBBUpper, totalPairs);
   ArrayResize(cachedBBLower, totalPairs); ArrayResize(cachedStochK, totalPairs); ArrayResize(cachedStochD, totalPairs);
   ArrayResize(cachedMACD, totalPairs); ArrayResize(cachedMACDSignal, totalPairs