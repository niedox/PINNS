:===IC Name = Kv1.4    ID:7862, DateTime [07-Apr-2019 23:49:50] Error [0.18]===
:m0 = 0.00, m1 = 1.00, h0 = 1.00, h1 = 1.00, mPower = 2, hPower = 1
:revPot = -96.20, startVolt = -40.00, endVolt = 50.00, zeroVolt = -50.00
:mInf  1/(1+exp((v- -17.5459)/-10.7706))
:mTau  sig2(v, -41.6945, 1.891, 22.7162, -37.9039, 7.6657, 0.98226, 11.4822, -18.1925, 5.0005)
:hInf  (1-0.97163) +( 0.97163 / (1 + exp((v - -35.2842)/5.7361)))
:hTau  19.6164 +( 298.2326 / (1 + exp((v - -23.5359)/5.0002)))
:==========End============

NEURON  {
	SUFFIX Kv1_4
	USEION k READ ek WRITE ik
	RANGE gKv1_4bar, gKv1_4, ik
}

UNITS   {
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER   {
	gKv1_4bar = 0.1 (S/cm2)
}

ASSIGNED    {
	v   (mV)
	ek  (mV)
	ik  (mA/cm2)
	gKv1_4  (S/cm2)
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
	gKv1_4 = gKv1_4bar *m*m*h
	ik = gKv1_4*(v-ek)
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
		mTau = sig2(v, -41.6945, 1.891, 22.7162, -37.9039, 7.6657, 0.98226, 11.4822, -18.1925, 5.0005)                    
		mInf = 1/(1+exp((v- -17.5459)/-10.7706))
		hInf = (1-0.97163) +( 0.97163 / (1 + exp((v - -35.2842)/5.7361)))
		hTau = 19.6164 +( 298.2326 / (1 + exp((v - -23.5359)/5.0002)))
	UNITSON
}

FUNCTION sig2(v, vBreak, offset1, amp1, vh1, slope1, offset2, amp2, vh2, slope2){
	LOCAL sigswitch
	sigswitch = 1/(1+exp((v-vBreak)/3.0))
	sig2 = (sigswitch*(offset1+(amp1)/(1+exp( (v- vh1)/-slope1 ) ))) +  ((1-sigswitch)*offset2+(amp2-offset2)/(1+exp((v- vh2)/slope2)))
}