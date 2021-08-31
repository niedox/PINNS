:===IC Name = Kv1.3    ID:1794, Error [0.15]===
: m0 = 0.00, m1 = 1.00, h0 = 1.00, h1 = 1.00, mPower = 2, hPower = 1
: revPot = -96.20, startVolt = -40.00, endVolt = 50.00, zeroVolt = -50.00
: mInf  1/(1+exp((v- -0.28502)/-11.3589))
: hInf  (1-0.94326) +( 0.94326 / (1 + exp((v - -11.0144)/13.7033)))
: mTau  sig2(v, -40.9235, 1.9677, 46.4403, -79.9999, 12.1326, 0.81495, 3.5072, -13.4832, 10.9858)
: hTau  94.2117 +( 454.9297 / (1 + exp((v - -10.0628)/9.6305)))
: ==========End============
:*******************************************

NEURON  {
	SUFFIX Kv1_3
	USEION k READ ek WRITE ik
	RANGE gKv1_3bar, gKv1_3, ik
}

UNITS   {
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER   {
	gKv1_3bar = 0.1 (S/cm2)
}

ASSIGNED    {
	v   (mV)
	ek  (mV)
	ik  (mA/cm2)
	gKv1_3  (S/cm2)
	mInf
	mTau
	hInf
	hTau
	celsius (degC)
	qtTau
	qthInf
}

STATE   {
	m
	h
}

BREAKPOINT  {
	SOLVE states METHOD cnexp
	gKv1_3 = gKv1_3bar *m*m*m*m*h
	ik = gKv1_3*(v-ek)
}

DERIVATIVE states   {
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
	UNITSOFF	
		mInf = 1/(1+exp((v- -0.28502)/-11.3589))
		mTau = sig2(v, -40.9235, 1.9677, 46.4403, -79.9999, 12.1326, 0.81495, 3.5072, -13.4832, 10.9858)
		hInf = (1-0.94326) +( 0.94326 / (1 + exp((v - -11.0144)/13.7033)))
		hTau = 94.2117 +( 454.9297 / (1 + exp((v - -10.0628)/9.6305)))
	UNITSON
}

FUNCTION sig2(v, vBreak, offset1, amp1, vh1, slope1, offset2, amp2, vh2, slope2){
	LOCAL sigswitch
	sigswitch = 1/(1+exp((v-vBreak)/3.0))
	sig2 = (sigswitch*(offset1+(amp1)/(1+exp( (v- vh1)/-slope1 ) ))) +  ((1-sigswitch)*offset2+(amp2-offset2)/(1+exp((v- vh2)/slope2)))
}