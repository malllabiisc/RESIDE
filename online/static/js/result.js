function sliceSize(dataNum, dataTotal) {
	return (dataNum / dataTotal) * 360;
}

function addSlice(sliceSize, pieElement, offset, sliceID, color) {
	$(pieElement).append("<div class='slice "+sliceID+"'><span></span></div>");
	var offset = offset - 1;
	var sizeRotation = -179 + sliceSize;
	$("."+sliceID).css({
		"transform": "rotate("+offset+"deg) translate3d(0,0,0)"
	});
	$("."+sliceID+" span").css({
		"transform"       : "rotate("+sizeRotation+"deg) translate3d(0,0,0)",
		"background-color": color
	});
}

function iterateSlices(sliceSize, pieElement, offset, dataCount, sliceCount, color) {
	var sliceID = "s"+dataCount+"-"+sliceCount;
	var maxSize = 179;
	if(sliceSize<=maxSize) {
		addSlice(sliceSize, pieElement, offset, sliceID, color);
	} else {
		addSlice(maxSize, pieElement, offset, sliceID, color);
		iterateSlices(sliceSize-maxSize, pieElement, offset+maxSize, dataCount, sliceCount+1, color);
	}
}

function createPie(dataElement, pieElement) {
	var listData = [];
	$(dataElement+" span").each(function() {
		listData.push(Number($(this).html()));
	});
	var listTotal = 0;
	for(var i=0; i<listData.length; i++) {
		listTotal += listData[i];
	}
	var offset = 0;
	var color = [
		"cornflowerblue", 
		"olivedrab", 
		"orange", 
		"tomato", 
		"crimson", 
		"purple", 
		"turquoise", 
		"forestgreen", 
		"navy", 
		"gray"
	];
	for(var i=0; i<listData.length; i++) {
		var size = sliceSize(listData[i], listTotal);
		iterateSlices(size, pieElement, offset, i, 0, color[i]);
		$(dataElement+" li:nth-child("+(i+1)+")").css("border-color", color[i]);
		offset += size;
	}
}

createPie(".pieID.legend", ".pieID.pie");

function sortByKey(toSort, key) {
	for (var i = 0; i < toSort.length; i++) {
		toSort[i] = [toSort[i], i];
	}

	toSort.sort(function(left, right) {
		a = left[0].getElementsByClassName(key)[0].innerText
		b = right[0].getElementsByClassName(key)[0].innerText
		console.log(a)
		console.log(b)
		return a < b ? -1 : 1;
	});

	toSort.sortIndices = [];
	for (var j = 0; j < toSort.length; j++) {
		toSort.sortIndices.push(toSort[j][1]);
		toSort[j] = toSort[j][0];
	}

	return toSort;
}

function sortTriples(key){
	all_elems = $('*#data'); 
	temp_elems = $('*#data');

	all_supp  = $('*#data_supp')
	temp_supp = $('*#data_supp')

	res_sort  = sortByKey(all_elems, key)
	sortInd   = res_sort.sortIndices

	console.log(sortInd.length)
	console.log(sortInd)

	var final_list = Array()
	for(i = 0; i < sortInd.length; i++){
		ind = sortInd[i]
		$('.table').append(temp_elems[ind])
		$('.table').append(temp_supp[ind])
	}

	// $('*#data').remove()
	// $('*#data_supp').remove()

	// console.log(final_list.length)
	// console.log(final_list)
	// return final_list
}