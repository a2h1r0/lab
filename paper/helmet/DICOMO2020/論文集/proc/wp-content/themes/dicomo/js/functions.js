$(function() {
	//ページトップへ戻るボタン
	var topBtn = $('#pageTop');
	topBtn.hide();
	$(window).scroll(function () {
		if ($(this).scrollTop() > 100) {
			topBtn.fadeIn();
		} else {
			topBtn.fadeOut();
		}
	});
    topBtn.click(function () {
		$('body,html').animate({
			scrollTop: 0
		}, 1000);
		return false;
    });
    //最新情報のカテゴリー切り替え
    $('ul.tab li:first a').addClass('selected');
    $('ul.panel li:not(:first)').hide();
    $('ul.tab li a').click(function(){
    	if(!$(this).hasClass('selected')){
    		$('ul.tab li a.selected').removeClass('selected');
    		$(this).addClass('selected');
    		$('ul.panel li').hide().filter($(this).attr('href')).fadeIn();
    	}
    	return false;
    });
    
    //ストライプテーブル
    $('#content .text table tr:odd').addClass('odd');
    $('#content .text table th:even').addClass('even');
    
    
});