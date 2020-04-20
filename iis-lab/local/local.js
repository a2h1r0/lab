function gate(){
   var UserInput = prompt("パスワードを入力して下さい:","");
   if( /\W+/g.test(UserInput) ) {
      alert("半角英数字のみを入力して下さい。");
   }
   else if( UserInput != null ) {
      location.href = UserInput + ".html";
   }
}
