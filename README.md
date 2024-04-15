# rust-threshold-signature
   what we need to do for changing the minimum threshold for a team to make a proposal or vote, threshold-signature can do that easily, you just need to change the public key of the group, and it is reusable if the threshold doesn't change. With this project we can detact who has been hack to send the wrong signature for the protocal, and even doesn't need a merkle tree to proof the member is in the group, next I will explame step by step from making a public key, do some secret exchange for the signature, how we verify the message for secret exchange, and detect who has been hack, and finish the signature, the idea comes from ````https://eprint.iacr.org/2020/540.pdf````.
   
## generate public key
  the team hold a ceremony to generate public key, each of them provide their secret key and a randum number to generate a polynomial, the secret is const and the degree can decide threshold. 
  The detail of math is the following, player i has secret:````si```` randum num:````ai````, he send ````si + ai*j + ai*j^2```` to player j (if degree is 2), so each play j has ````∑s + ∑a*j + ∑a*j^2````, and if three play do the lagrange they can calculate public key(in elliptic curve form) and secret(or we can call it private key, and they should not do that), src in ````key_generate.rs````.
  
## ecdsa
  after we have a public key, we can use ecdsa for verify the proposal, but before that they need to cooperate for ecdsa, for prevent someone monopolize other's information after he gets other's message, we use multiplication to add (MTA).
  The detail of math is the following, player has secret ````x```` , chose a randum num ````k````, publish public key ````x*G```` , commitment ````(k⁻¹)*G```` , response ````s = k * ( m + xr )```` (r is x coordinate of elliptic curve point commitment, m is message). Verifier can calculate ````s⁻¹(m*g) + s⁻¹(r*pub_key) === commitment````, src in ````other_small_project/ecdsa.rs````.

## multiplication to add (MTA)
  player1 has A, player2 has B, if they want to mut them without let other knows their secret, player1 send a cipher ````En(A)```` , player2 do mut and add to it ````En(A*B + C)```` (C is a randum num he chose), then player1 decrypt the cipher so he got ````A*B + C```` , player2 hold ````-C```` , so they add this together to get ````A * B```` , src in ````other_small_project/mta.rs````, and how can we make sure other sends the correct secret, I use zkp in ````paillier_verify.rs````.

## MTA message verify
  
