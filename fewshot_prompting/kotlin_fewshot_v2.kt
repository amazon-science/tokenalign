/**
 * You are an expert Kotlin programmer, and here is your task.
 * Write a function to locate the right insertion point for a specified value in sorted order.
 * >>> rightInsertion([1, 2, 4, 5], 6)
 * 4
 * >>> rightInsertion([1, 2, 4, 5], 3)
 * 2
 * >>> rightInsertion([1, 2, 4, 5], 7)
 * 4
 */
fun rightInsertion(a : List<Int>, x : Int) : Int {

// --------------------- COMPLETION (model generated) ---------------------
    var low = 0
    var high = a.size - 1
    while (low <= high) {
        var mid = (low + high) / 2
        if (a[mid] == x) {
            return mid
        } else if (a[mid] < x) {
            low = mid + 1
        } else {
            high = mid - 1
        }
    }
    return low
}

/**
 * You are an expert Kotlin programmer, and here is your task.
 * Write a function to perform the concatenation of two string tuples.
 * >>> concatenateStrings(["Manjeet", "Nikhil", "Akshat"], [" Singh", " Meherwal", " Garg"])
 * ["Manjeet Singh", "Nikhil Meherwal", "Akshat Garg"]
 * >>> concatenateStrings(["Shaik", "Ayesha", "Sanya"], [" Dawood", " Begum", " Singh"])
 * ["Shaik Dawood", "Ayesha Begum", "Sanya Singh"]
 * >>> concatenateStrings(["Harpreet", "Priyanka", "Muskan"], ["Kour", " Agarwal", "Sethi"])
 * ["HarpreetKour", "Priyanka Agarwal", "MuskanSethi"]
 */
fun concatenateStrings(testTup1 : List<String>, testTup2 : List<String>) : List<String> {

// --------------------- COMPLETION (model generated) ---------------------
    val result = mutableListOf<String>()
    for (i in 0 until testTup1.size) {
        result.add(testTup1[i] + testTup2[i])
    }
    return result
}