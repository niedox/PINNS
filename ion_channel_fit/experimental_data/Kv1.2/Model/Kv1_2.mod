:===IC Name = Kv1.2    ID:2505, DateTime [07-Apr-2019 23:37:22] Error [0.20]===
:m0 = 0.00, m1 = 1.00, h0 = 1.00, h1 = 1.00, mPower = 2, hPower = 1
:revPot = -96.20, startVolt = -40.00, endVolt = 50.00, zeroVolt = -50.00
:mInf  1/(1+exp((v- -8.1607)/-16.2041))
:hInf  (1-0.6669) +( 0.6669 / (1 + exp((v - -13.2501)/13.896)))
:mTau  sig2(v, -79.1345, 0.27482, 38.4251, -53.4992, 5.0003, 0.54781, 5.6426, -18.1111, 12.5306)
:hTau  99.0499 +( 421.1463 / (1 + exp((v - -10.7858)/13.1357)))
:==========End============

NEURON  {
	SUFFIX Kv1_2
	USEION k READ ek WRITE ik
	RANGE gKv1_2bar, gKv1_2, ik
}

UNITS   {
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER   {
	gKv1_2bar = 0.1 (S/cm2)
}

ASSIGNED    {
	v   (mV)
	ek  (mV)
	ik  (mA/cm2)
	gKv1_2  (S/cm2)
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
	gKv1_2 = gKv1_2bar *m*m*h
	ik = gKv1_2*(v-ek)
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
		mTau = sig2(v, -79.1345, 0.27482, 38.4251, -53.4992, 5.0003, 0.54781, 5.6426, -18.1111, 12.5306)
		mInf = 1/(1+exp((v- -8.1607)/-16.2041))
		hInf = (1-0.6669) +( 0.6669 / (1 + exp((v - -13.2501)/13.896)))
		hTau = 99.0499 +( 421.1463 / (1 + exp((v - -10.7858)/13.1357)))
	UNITSON
}

FUNCTION sig2(v, vBreak, offset1, amp1, vh1, slope1, offset2, amp2, vh2, slope2){
	LOCAL sigswitch
	sigswitch = 1/(1+exp((v-vBreak)/3.0))
	sig2 = (sigswitch*(offset1+(amp1)/(1+exp( (v- vh1)/-slope1 ) ))) +  ((1-sigswitch)*offset2+(amp2-offset2)/(1+exp((v- vh2)/slope2)))
}